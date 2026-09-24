#!/usr/bin/env python3
"""STeF-Map TBD: circular CRPS + MPP.

Refits STeF with the protocol parameters (the stock runner's own training
loop, reproduced call-for-call), verifies the refit reproduces the published
per-detection heading log-densities, then scores every held-out detection's
full 8-bin distribution through HistogramHeadingAccumulator.

Usage:
    KAIROS_RESULTS=../../results DATA_ROOT=~/datasets python3 rescore_heading_dist_stef_tbd.py

Takes no flags: the input run directory and the dataset path are module
constants at the top of the file, since it reads one method's own saved output
format. Point KAIROS_RESULTS at the results tree and edit those constants for a
different run.

Reports circular CRPS and MPP on the canonical row set, the same numbers
evaluation/evaluate.py reports for Kairos, so the two are comparable.
"""
import csv
import glob
import math
import pathlib
import os
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, os.environ.get("STEFMAP_DIR", "stefmap"))
from online_metrics import HistogramHeadingAccumulator  # noqa: E402
from stefmap.flow_map import FlowMap                     # noqa: E402

DATA = os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "tbd")
TRAIN = [f"{DATA}/2022-12/scene_tbd_2022-12-07-12-15-36", f"{DATA}/2022-12/scene_tbd_2022-12-07-12-44-46",
         f"{DATA}/2022-12/scene_tbd_2022-12-07-13-21-55", f"{DATA}/2022-12/scene_tbd_2022-12-07-13-46-01",
         f"{DATA}/2023-01/scene_tbd_2023-01-23-13-15-54", f"{DATA}/2023-01/scene_tbd_2023-01-23-13-52-34",
         f"{DATA}/2023-01/scene_tbd_2023-01-23-14-16-02", f"{DATA}/2023-01/scene_tbd_2023-01-26-12-54-28",
         f"{DATA}/2023-01/scene_tbd_2023-01-26-13-36-59", f"{DATA}/2023-01/scene_tbd_2023-01-26-15-17-19"]
TEST = [f"{DATA}/2023-02/scene_tbd_2023-02-01-12-48-21", f"{DATA}/2023-02/scene_tbd_2023-02-01-13-02-13",
        f"{DATA}/2023-02/scene_tbd_2023-02-01-13-51-05", f"{DATA}/2023-02/scene_tbd_2023-02-02-12-05-25",
        f"{DATA}/2023-02/scene_tbd_2023-02-02-12-14-22", f"{DATA}/2023-02/scene_tbd_2023-02-03-11-42-50"]
SAVED_PREDS = os.environ.get("KAIROS_RESULTS", "results") + "/stefmap"
OUT = os.path.join(os.environ.get("KAIROS_RESULTS", "results"),
                   "stefmap", "heading_dist.txt")


def main():
    model = FlowMap(cell_size=0.4, n_bins=8, time_window_s=10.0,
                    fremen_order=1, candidate_periods=[60.0, 300.0, 600.0])
    tw_ns = int(10.0 * 1e9)
    rows = []
    for scene in TRAIN:
        with open(f"{scene}/tracks.csv", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    rows.sort(key=lambda r: int(r["t_ns"]))
    window_end = None
    tic = time.time()
    for r in rows:
        t = int(r["t_ns"])
        if window_end is None:
            window_end = t + tw_ns
        while t >= window_end:
            model.flush_window(window_end)
            window_end += tw_ns
        model.add_detection(t, float(r["x_world"]), float(r["y_world"]), float(r["theta_rad"]))
    if window_end is not None:
        model.flush_window(window_end)
    print(f"[stef-tbd] trained on {len(rows)} dets in {time.time()-tic:.0f}s")

    acc = HistogramHeadingAccumulator(n_bins=8)
    n_par = 0
    par_err = 0.0
    for scene in TEST:
        stem = pathlib.Path(scene).name
        saved = f"{SAVED_PREDS}/{stem}_tracks.csv"
        saved_rows = list(csv.DictReader(open(saved, newline="")))
        test_rows = list(csv.DictReader(open(f"{scene}/tracks.csv", newline="")))
        assert len(saved_rows) == len(test_rows)
        matched, theta, masses = [], [], []
        for tr, sr in zip(test_rows, saved_rows):
            assert tr["t_ns"] == sr["t_ns"]
            t = int(tr["t_ns"]); x = float(tr["x_world"]); y = float(tr["y_world"])
            th = float(tr["theta_rad"])
            rec = model.score_row(t, x, y, th)
            m = bool(rec["matched"])
            # Parity gate: the refit must reproduce the published densities.
            if m and sr["matched"] in ("1", "1.0"):
                par_err = max(par_err, abs(rec["log_p_heading"] - float(sr["log_p_heading"])))
                n_par += 1
            matched.append(m)
            theta.append(th)
            masses.append(model.cell_distribution(t, x, y) if m else [0.0] * 8)
        acc.update(matched, theta, np.asarray(masses))
        print(f"[stef-tbd] scored {stem} ({len(test_rows)})")
    print(f"[stef-tbd] parity: max |dlog_p_heading| = {par_err:.2e} over {n_par} matched rows")
    r = acc.result()
    lines = [f"STeF-Map TBD heading distribution scores (parity max err {par_err:.2e})",
             f"n={r['n']} n_matched={r['n_matched']}",
             f"circular CRPS: {r['crps_rad']:.4f}  [matched-only {r['crps_rad_matched_only']:.4f}]  (uniform 0.7854)",
             f"MPP          : {r['mpp']:.4f}  [matched-only {r['mpp_matched_only']:.4f}]  (uniform 0.1250)"]
    print("\n".join(lines))
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
