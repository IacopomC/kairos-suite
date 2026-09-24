#!/usr/bin/env python3
"""Aion TBD: circular CRPS + MPP from the rescore container's output.

`preds_rescore.csv` = the 6 real test CSVs scored first (row-for-row parity
gate against the published preds.csv), then 8 bin-center queries per real
detection whose densities give the per-detection 8-bin masses.

Usage:
    KAIROS_RESULTS=../../results DATA_ROOT=~/datasets python3 rescore_heading_dist_aion_tbd.py

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
from online_metrics import HistogramHeadingAccumulator  # noqa: E402

R = os.environ.get("KAIROS_RESULTS", "results") + "/aion_perscene"
DATA = os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "tbd/2023-02")
TEST = ["scene_tbd_2023-02-01-12-48-21", "scene_tbd_2023-02-01-13-02-13",
        "scene_tbd_2023-02-01-13-51-05", "scene_tbd_2023-02-02-12-05-25",
        "scene_tbd_2023-02-02-12-14-22", "scene_tbd_2023-02-03-11-42-50"]


def main():
    theta = []
    for s in TEST:
        with open(f"{DATA}/{s}/tracks.csv", newline="") as f:
            theta.extend([float(r["theta_rad"]) for r in csv.DictReader(f)])
    n_real = len(theta)
    rows = list(csv.DictReader(open(f"{R}/preds_rescore.csv", newline="")))
    assert len(rows) == n_real + 8 * n_real, f"{len(rows)} != 9*{n_real}"
    pub = list(csv.DictReader(open(f"{R}/preds.csv", newline="")))
    par_err = 0.0
    n_par = 0
    matched = []
    for pr, pb in zip(rows[:n_real], pub):
        assert pr["t_ns"] == pb["t_ns"]
        m = pr["matched"] in ("1", "1.0")
        matched.append(m)
        if m and pb["matched"] in ("1", "1.0"):
            par_err = max(par_err, abs(float(pr["log_p_heading"]) - float(pb["log_p_heading"])))
            n_par += 1
    print(f"[aion-tbd] parity vs published preds: max |dlog_p_heading| = {par_err:.2e} over {n_par} rows")

    masses = np.zeros((n_real, 8))
    for i in range(n_real):
        for k in range(8):
            p = rows[n_real + 8 * i + k]
            lp = p["log_p_heading"]
            if p["matched"] in ("1", "1.0") and lp not in ("", "nan"):
                masses[i, k] = math.exp(float(lp)) * (math.pi / 4.0)
    acc = HistogramHeadingAccumulator(n_bins=8)
    for lo in range(0, n_real, 100000):
        hi = min(lo + 100000, n_real)
        acc.update(matched[lo:hi], theta[lo:hi], masses[lo:hi])
    r = acc.result()
    lines = [f"Aion TBD heading distribution scores (parity max err {par_err:.2e})",
             f"n={r['n']} n_matched={r['n_matched']}",
             f"circular CRPS: {r['crps_rad']:.4f}  [matched-only {r['crps_rad_matched_only']:.4f}]  (uniform 0.7854)",
             f"MPP          : {r['mpp']:.4f}  [matched-only {r['mpp_matched_only']:.4f}]  (uniform 0.1250)"]
    print("\n".join(lines))
    open(f"{R}/heading_dist.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
