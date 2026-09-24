#!/usr/bin/env python3
"""Run STeF-Map on ATC, online and streaming.

STeF-Map's FremenCell is a fixed-size running-DFT model (O(cells) memory, online
updates); the only batch step in the stock runner is loading+sorting every CSV.
The ATC per-day npz are already time-sorted and chronological, so streaming them
day by day reproduces the global-sort training bit-identically without holding
the corpus in memory. Heading-only: scored on the heading channel through the
same FlowAccumulator KAIROS uses, so the heading MLPD is directly comparable
(identical uniform-density floor and averaging). Periods are a CLI parameter,
defaulting to the ATC set.

Usage:
    python3 run_stefmap_atc.py --help

Needs the STeF-Map source tree; pass its location rather than relying on a
default. Trains and scores online in one pass, feeding the same accumulators as
every other method, so no per-detection files are written.
"""
import argparse
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                   # atc_stream
sys.path.insert(0, str(_HERE.parent / "evaluation"))            # online_metrics
sys.path.insert(0, "/ws/stefmap")                               # stefmap package
from atc_stream import AtcStream                                # noqa: E402
from online_metrics import FlowAccumulator, HistogramHeadingAccumulator  # noqa: E402
from stefmap.flow_map import FlowMap                            # noqa: E402


def _bounds(map_pgm, map_yaml):
    """In-bounds rectangle from the occupancy MAP extent (not the nav graph).
    STeF-Map is god's-eye and uses neither the robot nor the graph; this just
    keeps detections within the mapped area."""
    from PIL import Image
    res = origin = None
    with open(map_yaml) as f:
        for line in f:
            if line.startswith("resolution:"):
                res = float(line.split(":")[1])
            if line.startswith("origin:"):
                origin = [float(v) for v in line.split("[")[1].split("]")[0].split(",")]
    img = np.array(Image.open(map_pgm))
    H, W = img.shape[:2]
    x0, y0 = origin[0], origin[1]
    return ((x0, x0 + W * res), (y0, y0 + H * res))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-pgm", required=True, help="ATC occupancy PGM (for the in-bounds rectangle)")
    ap.add_argument("--map-yaml", required=True, help="ATC occupancy YAML (origin + resolution)")
    ap.add_argument("--train-days", nargs="+", required=True, help="chronological train npz")
    ap.add_argument("--test-days", nargs="+", required=True, help="held-out test npz")
    ap.add_argument("--cell-size", type=float, default=0.4)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--time-window-s", type=float, default=10.0)
    ap.add_argument("--fremen-order", type=int, default=1)
    ap.add_argument("--periods-s", type=float, nargs="+",
                    default=[3600.0, 43200.0, 86400.0, 604800.0],
                    help="FreMEn candidate periods (dataset parameter)")
    args = ap.parse_args()

    xb, yb = _bounds(args.map_pgm, args.map_yaml)
    model = FlowMap(cell_size=args.cell_size, n_bins=args.n_bins,
                    time_window_s=args.time_window_s, fremen_order=args.fremen_order,
                    candidate_periods=list(args.periods_s))
    tw_ns = int(args.time_window_s * 1e9)

    # ---- train: stream chronological days (== global sort, bit-identical) ----
    window_end_ns = None; n_train = 0; tic = time.time()
    for day in args.train_days:
        s = AtcStream(day, x_bounds=xb, y_bounds=yb)
        t_ns = (s.t * 1e9).astype(np.int64)
        x = s.x; y = s.y; th = s.theta
        for i in range(t_ns.size):
            ti = int(t_ns[i])
            if window_end_ns is None:
                window_end_ns = ti + tw_ns
            while ti >= window_end_ns:
                model.flush_window(window_end_ns)
                window_end_ns += tw_ns
            model.add_detection(ti, float(x[i]), float(y[i]), float(th[i]))
        n_train += t_ns.size
        print(f"  [train] {pathlib.Path(day).name}: {t_ns.size} dets "
              f"({n_train/(time.time()-tic):.0f}/s cum)", flush=True)
    if window_end_ns is not None:
        model.flush_window(window_end_ns)
    st = model.summary()
    print(f"[stefmap] trained on {n_train} dets, {st['n_cells']} active cells "
          f"in {(time.time()-tic)/60:.1f} min", flush=True)

    # ---- score: heading channel through KAIROS's FlowAccumulator, plus the
    # full 8-bin distribution through the circular CRPS / MPP accumulator ----
    flow = FlowAccumulator()
    hd = HistogramHeadingAccumulator(n_bins=args.n_bins)
    for day in args.test_days:
        s = AtcStream(day, x_bounds=xb, y_bounds=yb)
        t_ns = (s.t * 1e9).astype(np.int64)
        n = t_ns.size
        matched = np.zeros(n, dtype=bool); lph = np.empty(n)
        masses = np.zeros((n, args.n_bins))
        for i in range(n):
            rec = model.score_row(int(t_ns[i]), float(s.x[i]), float(s.y[i]), float(s.theta[i]))
            matched[i] = bool(rec["matched"]); lph[i] = rec["log_p_heading"]
            if matched[i]:
                masses[i] = model.cell_distribution(int(t_ns[i]), float(s.x[i]), float(s.y[i]))
        nan = np.full(n, np.nan)
        flow.update(matched, s.rho, {"heading": lph, "speed": nan, "joint": nan}, nan)
        hd.update(matched, s.theta, masses)
        print(f"  [score] {pathlib.Path(day).name}: {n} dets", flush=True)

    f = flow.result()
    h = hd.result()
    print("\n----- STeF-Map ATC (heading channel) -----")
    print(f"  n={f['n_total']}  match_rate={f['match_rate']:.4f}")
    print(f"  MLPD heading : {f['mlpd_heading']:.4f}  "
          f"[matched-only {f['mlpd_heading_matched_only']:.4f}]")
    print(f"  circular CRPS: {h['crps_rad']:.4f}  [matched-only {h['crps_rad_matched_only']:.4f}]  (uniform 0.7854)")
    print(f"  MPP          : {h['mpp']:.4f}  [matched-only {h['mpp_matched_only']:.4f}]  (uniform 0.1250)")


if __name__ == "__main__":
    main()
