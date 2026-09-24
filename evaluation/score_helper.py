"""
Drive pipeline.score_observations over a test tracks.csv and write a
per-detection predictions CSV.

Importable from a Kairos runner script after the training phase is done:

    from score_helper import score_csv   # evaluation/ on sys.path
    score_csv(pipeline,
              test_tracks="path/to/test_tracks.csv",
              output="path/to/predictions_fixed_speed.csv",
              horizon_s=0.0)

The pipeline must already be initialized and have processed at least one
map update (so the voxel grid exists) before scoring. The scoring call
itself does not mutate the model.
"""

import csv
import math
import os
from typing import Optional


def _t_range(path: str):
    """One streaming pass for [min, max] t_ns (used only for the holdout split).
    Reads row by row; never buffers the file."""
    t_min = t_max = None
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            t = int(r["t_ns"])
            if t_min is None or t < t_min:
                t_min = t
            if t_max is None or t > t_max:
                t_max = t
    return t_min, t_max


def _stream_chunks(path: str, batch_size: int, t_threshold, skipped):
    """Yield chunks of up to ``batch_size`` parsed rows as parallel lists.

    Streams the CSV (a `DictReader` iterator), so memory is bounded by one chunk
    regardless of file size -- this is what lets a full-day track file be scored
    without loading tens of millions of rows at once. Rows before ``t_threshold``
    (holdout split) are skipped, and ``skipped[0]`` counts them.
    """
    keys = ("t_s", "xyz", "theta", "rho", "t_ns", "ped", "x", "y")
    buf = {k: [] for k in keys}
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            t_ns = int(r["t_ns"])
            if t_threshold is not None and t_ns < t_threshold:
                skipped[0] += 1
                continue
            x = float(r["x_world"]); y = float(r["y_world"]); z = float(r["z_world"])
            th = float(r["theta_rad"]) % (2.0 * math.pi)
            sp = float(r["rho_mps"])
            buf["t_s"].append(t_ns / 1e9); buf["xyz"].append([x, y, z])
            buf["theta"].append(th); buf["rho"].append(sp)
            buf["t_ns"].append(t_ns); buf["ped"].append(int(r["ped_id"]))
            buf["x"].append(x); buf["y"].append(y)
            if len(buf["t_s"]) >= batch_size:
                yield buf
                buf = {k: [] for k in keys}
    if buf["t_s"]:
        yield buf


def score_csv(
    pipeline,
    test_tracks: str,
    output: str,
    horizon_s: float = 0.0,
    batch_size: Optional[int] = None,
    holdout_frac: float = 0.0,
) -> dict:
    """Run pipeline.score_observations over the test tracks and dump
    per-detection predictions to a CSV.

    Streams the input in chunks of ``batch_size`` rows (default 500k), so a
    full-day track file (tens of millions of detections) is scored with bounded
    memory. The output is byte-for-byte the same as scoring in one pass; for
    small scenes (one chunk) behaviour is unchanged.

    Output columns:
        t_ns, ped_id, x_world, y_world, theta_obs, rho_obs,
        matched, log_p_joint, log_p_heading, log_p_speed, mean_rho_pred,
        p_present_pred, flag, std_theta, std_rho,
        slot_weight_0..7

    When ``holdout_frac > 0``, only detections whose timestamp falls in the last
    ``holdout_frac`` of the CSV's [min_t_ns, max_t_ns] range are scored (a
    deterministic time-split for same-env sanity runs). 0.0 (default) scores
    every row.
    """
    if batch_size is None or batch_size <= 0:
        batch_size = 500_000  # bounded chunk: caps memory, streams the file

    t_threshold = None
    if holdout_frac > 0.0:
        if holdout_frac >= 1.0:
            raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
        t_min, t_max = _t_range(test_tracks)
        if t_min is None:
            raise RuntimeError(f"No rows in {test_tracks}")
        t_threshold = t_min + int((1.0 - holdout_frac) * (t_max - t_min))

    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _K_SLOTS = 8
    n_matched = n_unmatched = n_total = 0
    skipped = [0]
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_ns", "ped_id", "x_world", "y_world", "theta_obs", "rho_obs",
                "matched", "log_p_joint", "log_p_heading", "log_p_speed",
                "mean_rho_pred", "p_present_pred", "flag",
                "std_theta", "std_rho",
            ]
            + [f"slot_weight_{k}" for k in range(_K_SLOTS)]
        )
        for buf in _stream_chunks(test_tracks, batch_size, t_threshold, skipped):
            m = len(buf["t_s"])
            scores = pipeline.score_observations(
                timestamps_s=buf["t_s"],
                xyz_world=buf["xyz"],
                theta=buf["theta"],
                rho=buf["rho"],
                horizon_s=horizon_s,
            )
            matched = scores["matched"]
            log_p = scores["log_p_joint"]
            # Channel marginals p(theta)/p(rho); nan fallback for a pre-marginals
            # binding (evaluate_channels.py renders nan as "--").
            log_p_h = scores.get("log_p_heading", [float("nan")] * m)
            log_p_s = scores.get("log_p_speed", [float("nan")] * m)
            m_rh = scores["mean_rho"]
            pp = scores["p_present"]
            fg = scores["flag"]
            # Predictive standard deviations.
            std_theta = scores.get("std_theta", [0.0] * m)
            std_rho = scores.get("std_rho", [0.0] * m)
            # Per-slot mixing weights; empty list for unmatched, all-zeros on an
            # older binding.
            raw_sw = scores.get("slot_weights", [[] for _ in range(m)])
            for j in range(m):
                is_match = int(matched[j])
                if is_match:
                    n_matched += 1
                else:
                    n_unmatched += 1
                sw = list(raw_sw[j]) if raw_sw[j] else []
                sw = (sw + [0.0] * _K_SLOTS)[:_K_SLOTS]
                w.writerow(
                    [
                        buf["t_ns"][j], buf["ped"][j], buf["x"][j], buf["y"][j],
                        buf["theta"][j], buf["rho"][j], is_match, log_p[j],
                        log_p_h[j], log_p_s[j], m_rh[j], pp[j], fg[j],
                        std_theta[j], std_rho[j],
                    ]
                    + sw
                )
            n_total += m

    if n_total == 0:
        raise RuntimeError(
            f"No test rows scored in {test_tracks} (holdout_frac={holdout_frac})"
        )

    return {
        "predictions": output,
        "n_total": n_total,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "match_rate": n_matched / n_total if n_total else 0.0,
        "holdout_frac": holdout_frac,
        "n_skipped_holdout": skipped[0],
    }
