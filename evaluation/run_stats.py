"""
Runtime + memory snapshot helper for kairos eval runs.

Call once at the end of a training+scoring run (after `score_helper.score_csv`)
to dump aggregate flow-map state stats plus the current process's RSS into a
JSON sidecar that `evaluate.py` will pick up automatically.

Convention: the sidecar lives next to the predictions CSV with a `.stats.json`
suffix instead of `.csv` --- e.g. `preds_fixed_speed.csv` pairs with
`preds_fixed_speed.stats.json`. `evaluate.py` looks for that filename by
default and folds the runtime/memory columns into its output table.

Usage in a runner script:

    from run_stats import record_stats   # evaluation/ on sys.path
    record_stats(
        pipeline,
        predictions_csv="/path/to/preds_fixed_speed.csv",
        label="fixed-+-speed",
    )
"""

import json
import os
import time
from typing import Optional


def _stats_path_for(predictions_csv: str) -> str:
    if predictions_csv.endswith(".csv"):
        return predictions_csv[:-4] + ".stats.json"
    return predictions_csv + ".stats.json"


def _lc_events_path_for(predictions_csv: str) -> str:
    """Sidecar path for the LC events list. Used by evaluate.py --lc-events."""
    if predictions_csv.endswith(".csv"):
        return predictions_csv[:-4] + ".lc_events.json"
    return predictions_csv + ".lc_events.json"


def _coupling_path_for(predictions_csv: str) -> str:
    """Sidecar path for the EdgeCouplingMap JSON dump."""
    if predictions_csv.endswith(".csv"):
        return predictions_csv[:-4] + ".coupling.json"
    return predictions_csv + ".coupling.json"


_TIMING_COMPONENTS = (
    # FlowMap per-observation / per-event timers.
    "add_observation",
    "ensure_cells",
    "remap_cells",
    "run_maintenance",
    "sharing",
    # Module-level timers at their natural cadence: per-frame (process_frame,
    # coupling), per-event (place_summaries), per-query (score). ingest has no
    # own count -- its mean is ingest_total_us / process_frame_count, derived
    # downstream rather than here.
    "process_frame",
    "coupling",
    "place_summaries",
    "score",
)


def _summarize_timing(flow_timing: dict) -> dict:
    """Derive per-call mean (ms) for each component alongside the raw totals.

    The C++ side reports `<component>_count` and `<component>_total_us`; we
    derive `<component>_mean_ms = total_us / count / 1000` so downstream
    consumers (evaluate.py, paper tables) can quote per-tick component cost
    without re-deriving. count=0 collapses to None.
    """
    out = dict(flow_timing)
    for comp in _TIMING_COMPONENTS:
        count = flow_timing.get(f"{comp}_count", 0) or 0
        total_us = flow_timing.get(f"{comp}_total_us", 0) or 0
        if count > 0:
            out[f"{comp}_mean_ms"] = (total_us / count) / 1000.0
        else:
            out[f"{comp}_mean_ms"] = None
    return out


_STABILITY_BUCKETS = (
    ("sparse", 1, 10),
    ("medium", 10, 50),
    ("dense", 50, None),
)


def _summarize_stability(records: list) -> dict:
    """Aggregate per-cell stability records into coverage-stratified summary.

    Each record is {total_observations, n_obs_to_stable}. -1 in
    n_obs_to_stable means the cell never stabilised within its observation
    budget. For each coverage bucket we report:
      - count: number of cells in the bucket
      - frac_stable: fraction of cells that reached stability
      - median, p90: among cells that DID stabilise, the distribution of
        n_obs_to_stable. Both reported in observation count.
    The "all" bucket aggregates over all cells regardless of coverage.
    """
    if not records:
        return {}

    def _summary(subset: list) -> dict:
        n = len(subset)
        stable = [r["n_obs_to_stable"] for r in subset if r["n_obs_to_stable"] >= 0]
        if n == 0:
            return {"count": 0, "frac_stable": None, "median": None, "p90": None}
        frac = len(stable) / n if n > 0 else 0.0
        if not stable:
            return {"count": n, "frac_stable": frac, "median": None, "p90": None}
        s = sorted(stable)
        mid = s[len(s) // 2]
        p90_idx = max(0, int(round(0.9 * (len(s) - 1))))
        return {
            "count": n,
            "frac_stable": frac,
            "median": float(mid),
            "p90": float(s[p90_idx]),
        }

    out = {"all": _summary(records)}
    for name, lo, hi in _STABILITY_BUCKETS:
        if hi is None:
            subset = [r for r in records if r["total_observations"] >= lo]
        else:
            subset = [r for r in records
                      if lo <= r["total_observations"] < hi]
        out[name] = _summary(subset)
    return out


def _read_vmrss_bytes():
    """Read current VmRSS and VmSize from /proc/self/status (Linux, no extra deps)."""
    rss = vms = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmSize:"):
                    vms = int(line.split()[1]) * 1024
                if rss is not None and vms is not None:
                    break
    except Exception:
        pass
    return rss, vms


def record_stats(
    pipeline,
    predictions_csv: str,
    label: Optional[str] = None,
    extra: Optional[dict] = None,
    elapsed_s: Optional[float] = None,
) -> dict:
    """Snapshot runtime + memory and write a `<predictions>.stats.json` sidecar.

    Returns the dict that was written so the caller can log/inspect it.
    Memory is read from /proc/self/status (Linux, always available in Docker).
    elapsed_s is the wall-clock training loop time in seconds (optional).
    """
    rss_bytes, vms_bytes = _read_vmrss_bytes()
    peak_rss_bytes = None
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = int(ru.ru_maxrss)
        peak_rss_bytes = maxrss * 1024 if maxrss < 2**32 else maxrss
    except Exception:
        pass

    try:
        flow_stats = pipeline.flow_stats()
    except AttributeError:
        flow_stats = {}

    try:
        flow_timing = pipeline.flow_timing()
    except AttributeError:
        flow_timing = {}

    try:
        stability_records = list(pipeline.flow_stability())
    except AttributeError:
        stability_records = []

    try:
        lc_events = list(pipeline.lc_events())
    except AttributeError:
        lc_events = []

    record: dict = {
        "schema": "kairos_run_stats_v1",
        "timestamp": time.time(),
        "predictions_csv": os.path.abspath(predictions_csv),
        "label": label,
        "elapsed_s": elapsed_s,
        "memory": {
            "rss_bytes": rss_bytes,
            "vms_bytes": vms_bytes,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "flow_stats": dict(flow_stats) if flow_stats else {},
        "flow_timing": _summarize_timing(flow_timing) if flow_timing else {},
        "stability": _summarize_stability(stability_records),
        "extra": extra or {},
    }

    out_path = _stats_path_for(predictions_csv)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    # Emit the LC events list as its own sidecar so a scoring run can
    # feed it straight to evaluate.py --lc-events without re-parsing the
    # main stats record.
    lc_path = _lc_events_path_for(predictions_csv)
    record["lc_events_path"] = lc_path
    record["lc_events_count"] = len(lc_events)
    with open(lc_path, "w") as f:
        json.dump({"events": lc_events}, f, indent=2)

    # Emit EdgeCouplingMap state as its own sidecar. evaluate.py --coupling-json
    # reads this to compute the true coupled J-MLPD for the coupling metric.
    # Empty ("{}") when the pipeline has no coupling state.
    coupling_json_str = "{}"
    try:
        coupling_json_str = pipeline.flow_coupling_json()
    except AttributeError:
        pass
    coupling_path = _coupling_path_for(predictions_csv)
    record["coupling_path"] = coupling_path
    with open(coupling_path, "w") as f:
        f.write(coupling_json_str)

    return record


def load_stats(predictions_csv: str) -> Optional[dict]:
    """Read the `.stats.json` sidecar paired with the given predictions CSV.
    Returns None if no sidecar exists.
    """
    p = _stats_path_for(predictions_csv)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        record = json.load(f)
    return record
