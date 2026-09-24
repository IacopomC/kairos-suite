#!/usr/bin/env python3
"""Rheos TBD: circular CRPS + MPP from the rescore container's output.

`preds_rescore.csv` = the 6 real test CSVs scored first (row-for-row parity
gate against the published preds.csv), then the synthetic 0.2 m grid queries
(720 angles per cell) whose densities define each cell's heading marginal.

Usage:
    KAIROS_RESULTS=../../results DATA_ROOT=~/datasets python3 rescore_heading_dist_rheos_tbd.py

Takes no flags: the input run directory and the dataset path are module
constants at the top of the file, since it reads one method's own saved output
format. Point KAIROS_RESULTS at the results tree and edit those constants for a
different run.

Reports circular CRPS and MPP on the canonical row set, the same numbers
evaluation/evaluate.py reports for Kairos, so the two are comparable.
"""
import csv
import math
import pathlib
import os
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from online_metrics import GridHeadingScorer  # noqa: E402

R = os.environ.get("KAIROS_RESULTS", "results") + "/rheos_perscene"
DATA = os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "tbd/2023-02")
TEST = ["scene_tbd_2023-02-01-12-48-21", "scene_tbd_2023-02-01-13-02-13",
        "scene_tbd_2023-02-01-13-51-05", "scene_tbd_2023-02-02-12-05-25",
        "scene_tbd_2023-02-02-12-14-22", "scene_tbd_2023-02-03-11-42-50"]
G = 720


def main():
    real = []
    for s in TEST:
        with open(f"{DATA}/{s}/tracks.csv", newline="") as f:
            real.extend([(int(r["t_ns"]), float(r["x_world"]), float(r["y_world"]),
                          float(r["theta_rad"])) for r in csv.DictReader(f)])
    n_real = len(real)
    rows = list(csv.DictReader(open(f"{R}/preds_rescore.csv", newline="")))
    assert len(rows) >= n_real, f"{len(rows)} < {n_real}"
    pub = list(csv.DictReader(open(f"{R}/preds.csv", newline="")))
    par_err = 0.0
    n_par = 0
    for pr, pb in zip(rows[:n_real], pub):
        assert pr["t_ns"] == pb["t_ns"]
        if pr["matched"] in ("1", "1.0") and pb["matched"] in ("1", "1.0"):
            par_err = max(par_err, abs(float(pr["log_p_heading"]) - float(pb["log_p_heading"])))
            n_par += 1
    print(f"[rheos-tbd] parity vs published preds: max |dlog_p_heading| = {par_err:.2e} over {n_par} rows")

    gs = GridHeadingScorer(n_grid=G)
    grid = rows[n_real:]
    assert len(grid) % G == 0
    # Rheos preds rows do not echo positions; rebuild each query's cell from
    # the synthetic input file (identical row order).
    gt = list(csv.DictReader(open(f"{R}/grid_tracks.csv", newline="")))
    assert len(gt) == len(grid)
    dens = {}
    for q, p in zip(gt, grid):
        key = (math.floor(float(q["x_world"]) / 0.2), math.floor(float(q["y_world"]) / 0.2))
        if p["matched"] in ("1", "1.0"):
            dens.setdefault(key, []).append(math.exp(float(p["log_p_heading"])))
        else:
            dens.setdefault(key, []).append(0.0)
    n_cells = 0
    for key, d in dens.items():
        d = np.asarray(d)
        if d.sum() > 0:
            gs.add_distribution(key, d)
            n_cells += 1
    print(f"[rheos-tbd] {n_cells} cells with usable density (of {len(dens)})")

    keys = [(math.floor(x / 0.2), math.floor(y / 0.2)) for _, x, y, _ in real]
    matched = [r["matched"] in ("1", "1.0") for r in rows[:n_real]]
    theta = [th for _, _, _, th in real]
    crps, mpp = gs.score(keys, matched, theta)
    ma = np.asarray(matched, bool)
    lines = [f"Rheos TBD heading distribution scores (parity max err {par_err:.2e}; "
             f"density sampled at 0.2 m cell centres)",
             f"n={n_real} n_matched={int(ma.sum())}",
             f"circular CRPS: {crps.mean():.4f}  [matched-only {crps[ma].mean():.4f}]  (uniform 0.7854)",
             f"MPP          : {mpp.mean():.4f}  [matched-only {mpp[ma].mean():.4f}]  (uniform 0.1250)"]
    print("\n".join(lines))
    open(f"{R}/heading_dist.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
