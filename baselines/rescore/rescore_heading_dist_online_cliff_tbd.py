#!/usr/bin/env python3
r"""Fast Online CLiFF-map on TBD: circular CRPS + MPP.

Trains the online CLiFF-map on the SAME 10 TBD train scenes the batch CLiFF-map
baseline uses (each scene's tracks.csv fed as one online update batch), then
scores every held-out detection of the SAME 6 test scenes through
online_metrics.GridHeadingScorer with the shared uniform-heading floor -- the
exact protocol of rescore_heading_dist_cliff_tbd.py, with the online SWGMM in
place of cliffmap.grid.CliffMapGrid as the only change.

Runs in the online-mod venv. No re-scoring of a saved-preds reference exists for
the online map (unlike the batch baseline's parity check), so none is done.

Usage:
    python3 rescore_heading_dist_online_cliff_tbd.py \
        --out heading_dist_tbd.txt --n-train-scenes 10 --n-test-scenes 6

--max-rows-per-scene caps the work for a quick check; leave it off for the
reported numbers.
"""
import argparse
import csv
import math
import pathlib
import resource
import os
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from online_cliff_bridge import OnlineCliffModel, detections_to_online_csv  # noqa: E402
from online_cliff_report import (HeadingSums, new_flow_accumulator, report,  # noqa: E402
                                 write)

DATA = os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "tbd")
TRAIN = [f"{DATA}/2022-12/scene_tbd_2022-12-07-12-15-36", f"{DATA}/2022-12/scene_tbd_2022-12-07-12-44-46",
         f"{DATA}/2022-12/scene_tbd_2022-12-07-13-21-55", f"{DATA}/2022-12/scene_tbd_2022-12-07-13-46-01",
         f"{DATA}/2023-01/scene_tbd_2023-01-23-13-15-54", f"{DATA}/2023-01/scene_tbd_2023-01-23-13-52-34",
         f"{DATA}/2023-01/scene_tbd_2023-01-23-14-16-02", f"{DATA}/2023-01/scene_tbd_2023-01-26-12-54-28",
         f"{DATA}/2023-01/scene_tbd_2023-01-26-13-36-59", f"{DATA}/2023-01/scene_tbd_2023-01-26-15-17-19"]
TEST = [f"{DATA}/2023-02/scene_tbd_2023-02-01-12-48-21", f"{DATA}/2023-02/scene_tbd_2023-02-01-13-02-13",
        f"{DATA}/2023-02/scene_tbd_2023-02-01-13-51-05", f"{DATA}/2023-02/scene_tbd_2023-02-02-12-05-25",
        f"{DATA}/2023-02/scene_tbd_2023-02-02-12-14-22", f"{DATA}/2023-02/scene_tbd_2023-02-03-11-42-50"]
OUT_DEFAULT = os.path.join(os.environ.get("KAIROS_RESULTS", "results"),
                           "online_cliff", "heading_dist_tbd.txt")


def _read_scene(scene):
    t, x, y, th, rho = [], [], [], [], []
    with open(f"{scene}/tracks.csv", newline="") as f:
        for r in csv.DictReader(f):
            t.append(int(r["t_ns"]) / 1e9)
            x.append(float(r["x_world"])); y.append(float(r["y_world"]))
            th.append(float(r["theta_rad"])); rho.append(float(r["rho_mps"]))
    return (np.array(t), np.array(x), np.array(y), np.array(th), np.array(rho))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--n-train-scenes", type=int, default=0,
                    help="sanity slice: use only the first N train scenes (0 = all 10)")
    ap.add_argument("--n-test-scenes", type=int, default=0,
                    help="sanity slice: score only the first N test scenes (0 = all 6)")
    ap.add_argument("--max-rows-per-scene", type=int, default=0,
                    help="sanity slice: cap detections read per scene (0 = no cap). "
                         "FOR CODE-PATH CHECKS ONLY -- the real run uses 0.")
    args = ap.parse_args()

    params = {}
    if args.radius is not None:
        params["radius"] = args.radius
    if args.step is not None:
        params["step"] = args.step

    train = TRAIN[:args.n_train_scenes] if args.n_train_scenes else TRAIN
    test = TEST[:args.n_test_scenes] if args.n_test_scenes else TEST

    model = OnlineCliffModel(params=params, work_dir=args.work_dir)
    cap = args.max_rows_per_scene or None

    # --- adapter: each train scene -> online CLiFF-map 6-column CSV ---
    tic = time.time()
    batch_csvs = []
    n_train = 0
    for scene in train:
        t, x, y, th, rho = _read_scene(scene)
        if cap:
            t, x, y, th, rho = t[:cap], x[:cap], y[:cap], th[:cap], rho[:cap]
        out_csv = pathlib.Path(model.work_dir) / "train" / f"{pathlib.Path(scene).name}.csv"
        detections_to_online_csv(t, x, y, rho, th, out_csv)
        batch_csvs.append(str(out_csv))
        n_train += len(t)
    print(f"[online-cliff-tbd] adapter wrote {len(batch_csvs)} train CSVs "
          f"({n_train} dets) in {time.time()-tic:.0f}s", flush=True)

    # --- fit the online map, then reconstruct per-cell heading densities ---
    model.train(batch_csvs)
    gs = model.build_scorer()
    model.build_flow_model()
    print(f"[online-cliff-tbd] fitted {model.n_cells} cells "
          f"({model.n_cells_scored} scored) in {(time.time()-tic)/60:.1f} min "
          f"(radius={model.params['radius']} step={model.params['step']})", flush=True)

    head = HeadingSums()
    flow = new_flow_accumulator()
    for scene in test:
        t, x, y, th, rho = _read_scene(scene)
        if cap:
            t, x, y, th, rho = t[:cap], x[:cap], y[:cap], th[:cap], rho[:cap]
        s = model.score_flow(x, y, th, rho, gs)
        head.add(s["crps"], s["mpp"], s["matched"])
        flow.update(s["matched"], rho,
                    {"heading": s["log_p_heading"], "speed": s["log_p_speed"],
                     "joint": s["log_p_joint"]}, s["mean_rho_pred"])
        print(f"[online-cliff-tbd] scored {pathlib.Path(scene).name} "
              f"({len(th)}, matched {int(s['matched'].sum())})", flush=True)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    write(args.out, report("TBD (full visibility)", flow, head, peak_rss_mb,
                           extra=[f"cells fitted: {model.n_cells} "
                                  f"({model.n_cells_scored} scored)  "
                                  f"radius={model.params['radius']} step={model.params['step']}"]))
    model.cleanup()


if __name__ == "__main__":
    main()
