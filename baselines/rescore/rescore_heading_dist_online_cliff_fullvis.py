#!/usr/bin/env python3
r"""Fast Online CLiFF-map on ATC / HB in the FULL-VISIBILITY regime.

The embodied counterpart (rescore_heading_dist_online_cliff_embodied.py) trains
on the detections a simulated robot sees along its patrol. This script instead
trains on EVERY detection inside the scored rectangle, the same regime the batch
CLiFF-map and STeF-Map rows use (dagger in the flow table). Test scoring is
unchanged: every held-out detection in that rectangle, through the shared
GridHeadingScorer and FlowAccumulator floors.

Training days are streamed one at a time and their adapter CSV is deleted after
the update, because a full-visibility ATC day is ~17M detections (~1 GB of CSV)
and the corpus would otherwise need ~90 GB of scratch.

Reports every column of the main flow table: coverage, MLPD heading/speed/joint,
circular CRPS, MPP and Speed MAE. Runs in the online-mod venv.

Usage:
    python3 rescore_heading_dist_online_cliff_fullvis.py --source atc \
        --bounds <extract-dir>/bounds.json \
        --train-days <train...> --test-days <test...> \
        --out heading_dist_fullvis.txt

The full-visibility regime: trains on every detection in the scored rectangle
rather than only what a robot would have seen. Use it to separate what the
method costs from what partial observation costs.
"""
import argparse
import json
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
from online_cliff_bridge import OnlineCliffModel, detections_to_online_csv  # noqa: E402
from online_cliff_report import (HeadingSums, new_flow_accumulator, report,  # noqa: E402
                                 write)


def _open_day(source, day, xb, yb, hb_root, station, interp_hz, gap_max_s):
    """Whole-day stream, bounded by the scored rectangle. No field-of-view
    filter: this is the full-visibility regime."""
    if source == "atc":
        from atc_stream import AtcStream
        return AtcStream(day, x_bounds=xb, y_bounds=yb)
    from hb_stream import HBDayStream, hb_day_files
    files = hb_day_files(hb_root, station, day)
    if not files:
        raise SystemExit(f"[online-cliff-fv] no hb files for {day}")
    return HBDayStream(files, x_bounds=xb, y_bounds=yb,
                          interp_hz=interp_hz, gap_max_s=gap_max_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["atc", "hb"])
    ap.add_argument("--bounds", required=True,
                    help="bounds.json recording the scored rectangle + stream settings")
    ap.add_argument("--train-days", nargs="+", required=True)
    ap.add_argument("--test-days", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--x-bounds", type=float, nargs=2, default=None,
                    help="override the scored rectangle from --bounds. ATC's "
                         "full-visibility rows use the canonical occupancy-map "
                         "extent, which is wider than the embodied graph box.")
    ap.add_argument("--y-bounds", type=float, nargs=2, default=None)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--max-test-rows-per-day", type=int, default=0,
                    help="sanity slice: cap scored detections per test day (0 = all). "
                         "FOR CODE-PATH CHECKS ONLY.")
    ap.add_argument("--max-train-rows-per-day", type=int, default=0,
                    help="sanity slice: cap training detections per day (0 = all). "
                         "FOR CODE-PATH CHECKS ONLY -- the real run uses 0.")
    args = ap.parse_args()

    b = json.loads(pathlib.Path(args.bounds).read_text())
    xb = tuple(args.x_bounds) if args.x_bounds else tuple(b["x_bounds"])
    yb = tuple(args.y_bounds) if args.y_bounds else tuple(b["y_bounds"])
    station = b.get("station"); interp_hz = b.get("interp_hz")
    gap_max_s = b.get("gap_max_s")
    hb_root = b.get("hb_root", os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "hb-station/Data-2"))

    params = {}
    if args.radius is not None:
        params["radius"] = args.radius
    if args.step is not None:
        params["step"] = args.step
    model = OnlineCliffModel(params=params, work_dir=args.work_dir)
    train_dir = pathlib.Path(model.work_dir) / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    def produce(i):
        """Stream train day i in full and write it in the online CSV schema."""
        day = args.train_days[i]
        s = _open_day(args.source, day, xb, yb, hb_root, station,
                      interp_hz, gap_max_s)
        out_csv = train_dir / f"train_{pathlib.Path(day).stem}.csv"
        end = min(args.max_train_rows_per_day or len(s.t), len(s.t))
        pid = getattr(s, "id", None)
        detections_to_online_csv(s.t[:end], s.x[:end], s.y[:end], s.rho[:end],
                                 s.theta[:end], out_csv,
                                 person_id=None if pid is None else pid[:end])
        return str(out_csv)

    tic = time.time()
    model.train_lazy(produce, len(args.train_days))
    gs = model.build_scorer()
    model.build_flow_model()
    print(f"[online-cliff-fv] fitted {model.n_cells} cells "
          f"({model.n_cells_scored} scored) in {(time.time()-tic)/60:.1f} min "
          f"(radius={model.params['radius']} step={model.params['step']})", flush=True)

    head = HeadingSums()
    flow = new_flow_accumulator()
    for day in args.test_days:
        s = _open_day(args.source, day, xb, yb, hb_root, station,
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
        print(f"[online-cliff-fv] scored {pathlib.Path(day).stem} "
              f"(n_all so far {head.n}, matched {head.n_m})", flush=True)

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    label = "ATC" if args.source == "atc" else (station or "HB")
    write(args.out, report(f"{label} (full visibility)", flow, head, peak_rss_mb,
                           extra=[f"cells fitted: {model.n_cells} "
                                  f"({model.n_cells_scored} scored)  "
                                  f"radius={model.params['radius']} step={model.params['step']}",
                                  f"bounds x={xb} y={yb}"]))
    model.cleanup()


if __name__ == "__main__":
    main()
