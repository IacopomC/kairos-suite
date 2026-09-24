#!/usr/bin/env python3
"""CLiFF-map TBD: circular CRPS + MPP.

Refits CLiFF-map with the protocol parameters (the stock runner's training
path), verifies the refit reproduces the published per-detection heading
log-densities, then evaluates each fitted cell's heading marginal on the
720-point grid and scores every held-out detection through GridHeadingScorer.

Usage:
    KAIROS_RESULTS=../../results DATA_ROOT=~/datasets python3 rescore_heading_dist_cliff_tbd.py

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
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, os.environ.get("CLIFFMAP_DIR", "cliffmap"))
from online_metrics import GridHeadingScorer  # noqa: E402
from cliffmap.grid import CliffMapGrid        # noqa: E402

DATA = os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "tbd")
TRAIN = [f"{DATA}/2022-12/scene_tbd_2022-12-07-12-15-36", f"{DATA}/2022-12/scene_tbd_2022-12-07-12-44-46",
         f"{DATA}/2022-12/scene_tbd_2022-12-07-13-21-55", f"{DATA}/2022-12/scene_tbd_2022-12-07-13-46-01",
         f"{DATA}/2023-01/scene_tbd_2023-01-23-13-15-54", f"{DATA}/2023-01/scene_tbd_2023-01-23-13-52-34",
         f"{DATA}/2023-01/scene_tbd_2023-01-23-14-16-02", f"{DATA}/2023-01/scene_tbd_2023-01-26-12-54-28",
         f"{DATA}/2023-01/scene_tbd_2023-01-26-13-36-59", f"{DATA}/2023-01/scene_tbd_2023-01-26-15-17-19"]
TEST = [f"{DATA}/2023-02/scene_tbd_2023-02-01-12-48-21", f"{DATA}/2023-02/scene_tbd_2023-02-01-13-02-13",
        f"{DATA}/2023-02/scene_tbd_2023-02-01-13-51-05", f"{DATA}/2023-02/scene_tbd_2023-02-02-12-05-25",
        f"{DATA}/2023-02/scene_tbd_2023-02-02-12-14-22", f"{DATA}/2023-02/scene_tbd_2023-02-03-11-42-50"]
SAVED_PREDS = os.environ.get("KAIROS_RESULTS", "results") + "/cliffmap"
OUT = os.path.join(os.environ.get("KAIROS_RESULTS", "results"),
                   "cliffmap", "heading_dist.txt")
G = 720
THETA_G = np.arange(G) * (2.0 * math.pi / G)


def main():
    model = CliffMapGrid(cell_size=0.4, winding_number=1, min_velocity=0.1, max_clusters=50)
    n = 0
    tic = time.time()
    for scene in TRAIN:
        with open(f"{scene}/tracks.csv", newline="") as f:
            for r in csv.DictReader(f):
                model.add_detection(int(r["t_ns"]), float(r["x_world"]), float(r["y_world"]),
                                    float(r["theta_rad"]), float(r["rho_mps"]))
                n += 1
    print(f"[cliff-tbd] read {n} train dets in {time.time()-tic:.0f}s; fitting...", flush=True)
    model.fit(progress=True)
    print(f"[cliff-tbd] fit done in {(time.time()-tic)/60:.1f} min", flush=True)

    gs = GridHeadingScorer(n_grid=G)
    crps_sum = mpp_sum = crps_m = mpp_m = 0.0
    n_all = n_m = 0
    n_par = 0
    par_err = 0.0
    for scene in TEST:
        stem = pathlib.Path(scene).name
        saved_rows = list(csv.DictReader(open(f"{SAVED_PREDS}/{stem}_tracks.csv", newline="")))
        test_rows = list(csv.DictReader(open(f"{scene}/tracks.csv", newline="")))
        assert len(saved_rows) == len(test_rows)
        keys, matched, theta = [], [], []
        for tr, sr in zip(test_rows, saved_rows):
            assert tr["t_ns"] == sr["t_ns"]
            x = float(tr["x_world"]); y = float(tr["y_world"]); th = float(tr["theta_rad"])
            rec = model.score_row(x, y, th, float(tr["rho_mps"]))
            m = bool(rec["matched"])
            if m and sr["matched"] in ("1", "1.0"):
                par_err = max(par_err, abs(rec["log_p_heading"] - float(sr["log_p_heading"])))
                n_par += 1
            key = model._cell_key(x, y)
            if m and key not in gs._dists:
                cell = model._cells[key]
                dens = np.array([cell.heading_probability(t) for t in THETA_G])
                if dens.sum() <= 0:
                    m = False
                else:
                    gs.add_distribution(key, dens)
            keys.append(key); matched.append(m); theta.append(th)
        crps, mpp = gs.score(keys, matched, theta)
        ma = np.asarray(matched, bool)
        crps_sum += crps.sum(); mpp_sum += mpp.sum()
        crps_m += crps[ma].sum(); mpp_m += mpp[ma].sum()
        n_all += len(theta); n_m += int(ma.sum())
        print(f"[cliff-tbd] scored {stem} ({len(test_rows)})", flush=True)
    print(f"[cliff-tbd] parity: max |dlog_p_heading| = {par_err:.2e} over {n_par} matched rows")
    lines = [f"CLiFF-map TBD heading distribution scores (parity max err {par_err:.2e})",
             f"n={n_all} n_matched={n_m}",
             f"circular CRPS: {crps_sum/n_all:.4f}  [matched-only {crps_m/n_m:.4f}]  (uniform 0.7854)",
             f"MPP          : {mpp_sum/n_all:.4f}  [matched-only {mpp_m/n_m:.4f}]  (uniform 0.1250)"]
    print("\n".join(lines))
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
