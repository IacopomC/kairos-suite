#!/usr/bin/env python3
r"""Stage B (embodied): Fast Online CLiFF-map on ATC / HB -- circular CRPS + MPP.

Consumes the robot field-of-view TRAINING stream dumped by
extract_fov_stream_online_cliff.py (Stage A, in Docker) -- the identical stream
the rheos/aion ATC/HB baselines train on -- feeds each day as one online update
batch, then scores EVERY held-out detection of the test days through
online_metrics.GridHeadingScorer with the shared uniform-heading floor. The test
detections are the full held-out stream within the SAME detection rectangle
Stage A recorded in bounds.json (graph box pos +/- 6 m), exactly as the rheos
runner scores. Only the map construction (the online SWGMM) differs from the
batch CLiFF-map protocol.

Runs in the online-mod venv. Reads the test days with the benchmark's own
atc_stream / hb_stream (numpy / pandas only; no spark_dsg).

Usage:
    python3 rescore_heading_dist_online_cliff_embodied.py \
        --extract-dir <dir written by extract_fov_stream_online_cliff.py> \
        --test-days <held-out days...> --out heading_dist_embodied.txt

Stage B of two. Stage A (extract_fov_stream_online_cliff.py) dumps the robot's
field-of-view training stream; this builds the model online over it and scores
the held-out days.
"""
import argparse
import json
import math
import pathlib
import resource
import os
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
GS = _HERE.parent / "grounded_sim"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(GS))
from online_cliff_bridge import OnlineCliffModel  # noqa: E402
from online_cliff_report import (HeadingSums, new_flow_accumulator, report,  # noqa: E402
                                 write)


def _open_test_stream(source, day, xb, yb, hb_root, station, interp_hz, gap_max_s):
    if source == "atc":
        from atc_stream import AtcStream
        return AtcStream(day, x_bounds=xb, y_bounds=yb)
    if source == "tbd":
        from tbd_stream import TbdStream
        return TbdStream(f"{day}/tracks.csv", x_bounds=xb, y_bounds=yb)
    from hb_stream import HBDayStream, hb_day_files
    files = hb_day_files(hb_root, station, day)
    if not files:
        raise SystemExit(f"[online-cliff-emb] no hb files for {day}")
    return HBDayStream(files, x_bounds=xb, y_bounds=yb,
                          interp_hz=interp_hz, gap_max_s=gap_max_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", required=True,
                    help="Stage A output dir (bounds.json + train_*.csv)")
    ap.add_argument("--test-days", nargs="+", required=True,
                    help="ATC npz paths, or hb ISO dates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--max-test-rows-per-day", type=int, default=0,
                    help="sanity slice: cap scored detections per test day (0 = all). "
                         "FOR CODE-PATH CHECKS ONLY.")
    args = ap.parse_args()

    ext = pathlib.Path(args.extract_dir)
    bounds = json.loads((ext / "bounds.json").read_text())
    xb = tuple(bounds["x_bounds"]); yb = tuple(bounds["y_bounds"])
    source = bounds["stream_source"]; station = bounds["station"]
    interp_hz = bounds["interp_hz"]; gap_max_s = bounds["gap_max_s"]
    hb_root = bounds.get("hb_root", os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "hb-station/Data-2"))
    manifest = json.loads((ext / "train_manifest.json").read_text())
    batch_csvs = [str(ext / name) for name in manifest]

    params = {}
    if args.radius is not None:
        params["radius"] = args.radius
    if args.step is not None:
        params["step"] = args.step

    # --- fit the online map on the FoV training stream (one batch per day) ---
    tic = time.time()
    model = OnlineCliffModel(params=params, work_dir=args.work_dir)
    model.train(batch_csvs)
    gs = model.build_scorer()
    model.build_flow_model()
    print(f"[online-cliff-emb] fitted {model.n_cells} cells "
          f"({model.n_cells_scored} scored) in {(time.time()-tic)/60:.1f} min "
          f"(radius={model.params['radius']} step={model.params['step']})", flush=True)

    # --- score every held-out detection within the recorded bounds ---
    head = HeadingSums()
    flow = new_flow_accumulator()
    for day in args.test_days:
        s = _open_test_stream(source, day, xb, yb, hb_root, station,
                              interp_hz, gap_max_s)
        cap = args.max_test_rows_per_day or len(s.t)
        end = min(cap, len(s.t))
        for lo in range(0, end, args.chunk):
            hi = min(lo + args.chunk, end)
            rho = s.rho[lo:hi]
            sc = model.score_flow(s.x[lo:hi], s.y[lo:hi], s.theta[lo:hi], rho, gs)
            head.add(sc["crps"], sc["mpp"], sc["matched"])
            flow.update(sc["matched"], rho,
                        {"heading": sc["log_p_heading"], "speed": sc["log_p_speed"],
                         "joint": sc["log_p_joint"]}, sc["mean_rho_pred"])
        print(f"[online-cliff-emb] scored {pathlib.Path(day).stem} "
              f"(n_all so far {head.n}, matched {head.n_m})", flush=True)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    write(args.out, report(f"{source.upper()} (embodied path)", flow, head, peak_rss_mb,
                           extra=[f"cells fitted: {model.n_cells} "
                                  f"({model.n_cells_scored} scored)  "
                                  f"radius={model.params['radius']} step={model.params['step']}"]))
    model.cleanup()


if __name__ == "__main__":
    main()
