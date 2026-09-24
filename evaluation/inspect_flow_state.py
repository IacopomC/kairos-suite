#!/usr/bin/env python3
"""
Inspect a saved kairos flow-state binary (msgpack format).

The C++ side writes `flow_state.bin` either automatically at end of run
(via FlowTemporalModule::save) or on demand (via
pipeline.save_flow_state(path)). This script unpacks that file and prints
a high-level summary so you can verify the save is intact without having
to spin up the C++ pipeline.

What it reports:
    - Schema version + voxel size
    - Cell count, total components, total NUDFT models
    - Distribution of cells by component count
    - Sample per-cell numbers: gamma_0 means/stddevs across cells, observation
      counts
    - Bytes on disk + average bytes per cell

Usage:
    python3 inspect_flow_state.py /path/to/flow_state.bin
"""

import argparse
import os
import sys
from collections import Counter
from statistics import mean, median, stdev

try:
    import msgpack  # pip install msgpack
except ImportError:
    print("ERROR: msgpack required. Install with: pip install msgpack")
    sys.exit(1)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _safe_stats(values):
    if not values:
        return "(no samples)"
    if len(values) == 1:
        return f"n=1, value={values[0]:.4f}"
    return (
        f"n={len(values)}, mean={mean(values):+.4f}, "
        f"median={median(values):+.4f}, stdev={stdev(values):.4f}, "
        f"min={min(values):+.4f}, max={max(values):+.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a summary of a kairos flow_state.bin file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to flow_state.bin")
    parser.add_argument(
        "--max-cells-detailed",
        type=int,
        default=5,
        help="How many individual cells to print in detail (for sanity).",
    )
    parser.add_argument(
        "--gate-stats",
        action="store_true",
        help="Report the prequential order gate's per-cell decisions: for the "
             "weights and presence spectral models, the fraction of cells whose "
             "validated best order is >0 (harmonics engaged), the order "
             "histogram, and the observation-weighted share. This is the "
             "'which cells validated a rhythm' statistic.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: file not found: {args.path}")
        sys.exit(1)

    file_size = os.path.getsize(args.path)
    with open(args.path, "rb") as f:
        record = msgpack.unpack(f, raw=False, strict_map_key=False)

    print("=" * 60)
    print(f"FLOW STATE: {args.path}")
    print("=" * 60)
    print(f"File size:      {_fmt_bytes(file_size)} ({file_size:,} bytes)")
    print(f"Schema:         {record.get('schema', '(missing)')}")
    print(f"Version:        {record.get('version', '(missing)')}")
    print(f"Voxel size:     {record.get('voxel_size', '(missing)')} m")
    print(f"Cell count:     {record.get('cell_count', '(missing)')}")

    flow_map = record.get("flow_map")
    if not isinstance(flow_map, dict):
        print("ERROR: flow_map record is missing or malformed.")
        sys.exit(1)

    cells = flow_map.get("cells", [])
    if not cells:
        print("\nWARNING: zero cells in this save. Either nothing was ingested or"
              " the save fired before any cells were instantiated.")
        sys.exit(0)

    print(f"\nFlowMap version:  {flow_map.get('version')}")
    fm_cfg = flow_map.get("config", {})
    if fm_cfg:
        print(f"FlowMap config:   "
              f"nudft_min_obs={fm_cfg.get('nudft_min_obs')}")

    if args.gate_stats:
        min_obs_default = int(fm_cfg.get("nudft_min_obs", 1) or 1)

        def best_order(nd):
            ge = nd.get("gate_err") or []
            gn = int(nd.get("gate_n", 0) or 0)
            mo = int(nd.get("min_observations", min_obs_default) or 1)
            if gn < mo or not ge:
                return 0
            return int(min(range(len(ge)), key=lambda o: (ge[o], o)))

        def gate_pass(models, weight):
            # per cell: harmonics engaged if ANY of its models validates order>0
            orders = [best_order(nd) for nd in models if isinstance(nd, dict)]
            if not orders:
                return None
            return max(orders), Counter(orders)

        w_open = w_tot = 0
        w_open_obs = w_obs = 0
        p_open = p_tot = 0
        order_hist_w = Counter()
        order_hist_p = Counter()
        for cell in cells:
            obs = int(cell.get("total_observations", 0) or 0)
            wm = cell.get("nudft_models") or []
            gp = gate_pass(wm, obs)
            if gp is not None:
                mo_, hist = gp
                w_tot += 1
                w_obs += obs
                order_hist_w[mo_] += 1
                if mo_ > 0:
                    w_open += 1
                    w_open_obs += obs
            poisson = cell.get("poisson") or {}
            pn = poisson.get("nudft")
            if isinstance(pn, dict):
                bo = best_order(pn)
                p_tot += 1
                order_hist_p[bo] += 1
                if bo > 0:
                    p_open += 1
        print("\n" + "=" * 60)
        print("PREQUENTIAL ORDER GATE")
        print("=" * 60)
        if w_tot:
            print(f"Weights: gate open (any slot order>0) in {w_open}/{w_tot} cells "
                  f"({100.0 * w_open / w_tot:.1f} %), carrying "
                  f"{100.0 * w_open_obs / max(w_obs, 1):.1f} % of observations")
            print(f"  per-cell max validated order histogram: "
                  f"{dict(sorted(order_hist_w.items()))}")
        else:
            print("Weights: no gate accumulators found in this state")
        if p_tot:
            print(f"Presence: gate open in {p_open}/{p_tot} cells "
                  f"({100.0 * p_open / p_tot:.1f} %)")
            print(f"  validated order histogram: {dict(sorted(order_hist_p.items()))}")

    n_cells = len(cells)
    component_counts = []
    nudft_n = []
    nudft_gamma0 = []
    total_obs = []
    initialized_count = 0
    nudft_valid_count = 0
    has_original_set = 0

    for cell in cells:
        comps = cell.get("components") or []
        component_counts.append(len(comps))
        if cell.get("initialized"):
            initialized_count += 1
        if cell.get("nudft_valid"):
            nudft_valid_count += 1
        if cell.get("original_set"):
            has_original_set += 1
        total_obs.append(int(cell.get("total_observations", 0)))
        for m in (cell.get("nudft_models") or []):
            if isinstance(m, dict):
                nudft_n.append(int(m.get("n", 0)))
                nudft_gamma0.append(float(m.get("gamma0", m.get("gamma_0", 0.0))))

    print(f"\nCells:               {n_cells}")
    print(f"  initialized:       {initialized_count}  "
          f"({100.0 * initialized_count / n_cells:.1f}%)")
    print(f"  nudft_valid:       {nudft_valid_count}  "
          f"({100.0 * nudft_valid_count / n_cells:.1f}%)")
    print(f"  original_set:      {has_original_set}  "
          f"({100.0 * has_original_set / n_cells:.1f}%)")

    bucketed = Counter(component_counts)
    print("\nComponents per cell (histogram):")
    for k in sorted(bucketed):
        print(f"  K={k:2d}: {bucketed[k]} cells")

    print(f"\nTotal pi-channel NUDFTs serialised: {len(nudft_n)}")
    if nudft_n:
        print(f"  observation counts: {_safe_stats(nudft_n)}")
        print(f"  gamma_0 values:     {_safe_stats(nudft_gamma0)}")

    print(f"\nPer-cell total_obs:       {_safe_stats(total_obs)}")

    avg_per_cell = file_size / max(n_cells, 1)
    print(f"\nAverage on-disk per cell: {_fmt_bytes(int(avg_per_cell))}")

    if args.max_cells_detailed > 0:
        print(f"\nFirst {min(args.max_cells_detailed, n_cells)} cells in detail:")
        for cell in cells[: args.max_cells_detailed]:
            idx = cell.get("index", [None, None, None])
            comps = cell.get("components") or []
            pi_models = cell.get("nudft_models") or []
            print(f"  index={idx} K={len(comps)} "
                  f"total_obs={cell.get('total_observations')} "
                  f"initialized={cell.get('initialized')} "
                  f"nudft_valid={cell.get('nudft_valid')}")
            for k, m in enumerate(pi_models):
                if not isinstance(m, dict):
                    continue
                n_obs = m.get("n", 0)
                g0 = m.get("gamma0", m.get("gamma_0", 0.0))
                ng = len(m.get("gammas") or [])
                print(f"    pi NUDFT[{k}]: n={n_obs} gamma_0={g0:+.4f} F={ng}")


if __name__ == "__main__":
    main()
