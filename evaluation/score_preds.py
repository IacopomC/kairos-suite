#!/usr/bin/env python3
r"""Aggregate a kairos preds.csv into the full online metric set, WITHOUT rerunning
the model. The train+score pass already dumped one row per detection carrying
every quantity the metrics consume (matched, log_p_{heading,speed,joint},
theta_obs, rho_obs, mean_rho_pred, std_rho, x/y_world, ped_id, slot_weight_0..7);
this feeds those rows through the SAME accumulators evaluate.py's online loop uses
(online_metrics.{FlowAccumulator, CoverageStratifiedAccumulator,
HeadingDistributionAccumulator, CalibrationAccumulator, CouplingAccumulator}),
in the same order, so the output matches metrics_online.txt field-for-field.

Runs scored via the score_csv (preds-dump) path skip this aggregation; this
closes that gap for the dumped preds.

    python3 score_preds.py --preds results/tbd/preds.csv \
        --coverage-thresholds 10 50 [--coupling-json <preds.coupling.json>]
"""
import argparse
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from online_metrics import (FlowAccumulator, CoverageStratifiedAccumulator,  # noqa: E402
                            HeadingDistributionAccumulator, CalibrationAccumulator,
                            CouplingAccumulator, DEFAULT_SPEED_CAP_MPS)
from evaluate import (_print_flow, _print_cov, _print_heading_dist,  # noqa: E402
                      _print_cal, _print_coupling)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--coverage-thresholds", type=int, nargs=2, default=[10, 50],
                    help="per-voxel detection-count terciles (TBD 10 50, ATC 5000 20000)")
    ap.add_argument("--coupling-json", default=None,
                    help="preds.coupling.json for the coupling metric (edges "
                         "are byte-identical across a run's scenes)")
    ap.add_argument("--coupling-window-s", type=float, default=1.0)
    ap.add_argument("--voxel-size-m", type=float, default=0.4)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--speed-cap-mps", type=float, default=DEFAULT_SPEED_CAP_MPS)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()

    cols = open(args.preds).readline().strip().split(",")
    ix = {c: i for i, c in enumerate(cols)}
    sw_cols = [ix[f"slot_weight_{k}"] for k in range(8)]
    need = ["t_ns", "ped_id", "x_world", "y_world", "theta_obs", "rho_obs",
            "matched", "log_p_heading", "log_p_speed", "log_p_joint",
            "mean_rho_pred", "std_rho"]
    use = [ix[c] for c in need] + sw_cols
    d = np.loadtxt(args.preds, delimiter=",", skiprows=1, usecols=use)
    g = {c: d[:, i] for i, c in enumerate(need)}
    sw = d[:, len(need):]                       # (n, 8) slot weights
    t_s = g["t_ns"] / 1e9
    ids = g["ped_id"].astype(np.int64)
    xyz = np.column_stack([g["x_world"], g["y_world"], np.zeros(len(d))])
    theta, rho = g["theta_obs"], g["rho_obs"]
    matched = g["matched"] > 0.5
    log_p = {"heading": g["log_p_heading"], "speed": g["log_p_speed"],
             "joint": g["log_p_joint"]}
    mean_rho, std_rho = g["mean_rho_pred"], g["std_rho"]

    flow = FlowAccumulator(speed_cap_mps=args.speed_cap_mps)
    cov = CoverageStratifiedAccumulator(voxel_size=args.voxel_size_m,
                                        thresholds=tuple(args.coverage_thresholds))
    hd = HeadingDistributionAccumulator()
    cal = CalibrationAccumulator(n_bins=args.n_bins)
    cpl = None
    if args.coupling_json:
        cpl = CouplingAccumulator(open(args.coupling_json).read(),
                                  voxel_size=args.voxel_size_m,
                                  time_window_s=args.coupling_window_s)
        if not cpl.edge_lookup:
            print("[score_preds] coupling json has no edges for this graph; "
                  "skipping coupling block")
            cpl = None

    # Same chunked feed as evaluate.py's online loop. Order is irrelevant to
    # flow/cov/hd/cal (they are sums), but the coupling sliding window needs
    # non-decreasing time, so we break chunks at scene boundaries (t resets).
    resets = list(np.where(np.diff(t_s) < 0)[0] + 1)
    print(f"[score_preds] {len(d)} rows, matched {matched.mean():.4f}, "
          f"{len(resets)} scene boundaries")
    start = 0
    for b in resets + [len(t_s)]:
        for i in range(start, b, args.chunk):
            j = min(i + args.chunk, b)
            sl = slice(i, j)
            flow.update(matched[sl], rho[sl],
                        {c: log_p[c][sl] for c in log_p}, mean_rho[sl])
            cov.update(matched[sl], xyz[sl], log_p["joint"][sl])
            hd.update(matched[sl], theta[sl], sw[sl])
            cal.update(matched[sl], rho[sl], mean_rho[sl], std_rho[sl])
            if cpl is not None:
                cpl.update(matched[sl], t_s[sl], xyz[sl], theta[sl], rho[sl],
                           sw[sl], ids=ids[sl])
        start = b

    import io
    buf = io.StringIO()
    _real = sys.stdout
    sys.stdout = _Tee(_real, buf)
    try:
        print("----- FLOW (coverage-aware channel MLPD) -----"); _print_flow(flow.result())
        print("----- COVERAGE-STRATIFIED MLPD (joint) -----"); _print_cov(cov.result())
        print("----- HEADING DISTRIBUTION (circular CRPS / MPP) -----")
        _print_heading_dist(hd.result())
        print("----- CALIBRATION (speed uncertainty) -----"); _print_cal(cal.result())
        if cpl is not None:
            print("----- COUPLING (adjacent-voxel pairs, KL-stratified) -----")
            _print_coupling(cpl.result())
    finally:
        sys.stdout = _real
    if args.out:
        pathlib.Path(args.out).write_text(buf.getvalue())
        print(f"[score_preds] wrote {args.out}")


class _Tee:
    def __init__(self, *streams):
        self._s = streams

    def write(self, x):
        for s in self._s:
            s.write(x)

    def flush(self):
        for s in self._s:
            s.flush()


if __name__ == "__main__":
    main()
