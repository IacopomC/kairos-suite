#!/usr/bin/env python3
r"""Rheos baseline on the ATC simulation (train + online-score, no preds files).

Rheos is the online scene-graph SW-GMM baseline; on ATC it is the scalable
stand-in for the batch CLiFF-map (same SW-GMM, fit incrementally). No
reconstruction frontend: the simulated robot patrol supplies the place layer
(the given walkable graph, revealed as it explores) and its pose, exactly as
train_atc.py does for Kairos, but detections go to RheosCore.update_places /
add_detection. After fitting, the live model is scored over the held-out day(s)
and the flow metrics are accumulated in one streaming pass through the SHARED
online_metrics accumulators -- the same metric math evaluate.py uses -- so
nothing per-detection is written. Rheos carries no predictive variance, so only
the flow channels (heading/speed/joint MLPD, coverage, speed MAE) are reported.

Runs in the rheos Docker image.

Usage:
    python3 run_rheos_atc.py --graph <atc_graph.json> --days <train npz...> \
        --test-days <sep npz...> --out-dir results/atc_rheos
"""
import argparse
import math
import pathlib
import sys
import time
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evaluation"))
from robot_sim import (load_nav_graph, build_coverage_loop, RobotSim,
                       KEYFRAME_DT_S, ROBOT_SPEED_MPS, ACTIVE_WINDOW_RADIUS_M)
from atc_stream import AtcStream
from online_metrics import FlowAccumulator

from rheos_python import RheosCore
from rheos_python.incremental_runner import (
    build_config, load_model_params, DEFAULT_CONFIG_YAML, _predict_rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--days", nargs="+", required=True, help="train-day npz/csv")
    ap.add_argument("--test-days", nargs="+", required=True, help="held-out test-day npz/csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--assoc-radius", type=float, default=0.7)
    ap.add_argument("--grid-size", type=float, default=0.4)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cfg = build_config(load_model_params(DEFAULT_CONFIG_YAML),
                       overrides={"assoc_radius": args.assoc_radius, "grid_size": args.grid_size})
    model = RheosCore(cfg)

    pos, node_ids, adj, trav = load_nav_graph(args.graph)
    poly, _ = build_coverage_loop(pos, adj, trav)
    sim = RobotSim(poly, pos)
    xb = (pos[:, 0].min() - 6, pos[:, 0].max() + 6)
    yb = (pos[:, 1].min() - 6, pos[:, 1].max() + 6)
    r2 = ACTIVE_WINDOW_RADIUS_M ** 2
    print(f"[rheos_atc] {len(pos)} places, coverage loop {sim.total:.0f} m, {len(args.days)} train days")

    covered = set()
    for day in args.days:
        s = AtcStream(day, x_bounds=xb, y_bounds=yb)
        t0 = s.t_min
        dur = (s.t_max - t0) if args.minutes == 0 else args.minutes * 60.0
        n_kf = int(dur / KEYFRAME_DT_S)
        tic = time.time(); pushed = 0
        for k in range(n_kf):
            t = t0 + k * KEYFRAME_DT_S
            pose = sim.pose_at_arclength(ROBOT_SPEED_MPS * k * KEYFRAME_DT_S)
            vis = sim.visible_places(pose)
            new = [i for i in vis if i not in covered]
            if new:
                covered.update(new)
                model.update_places([(int(node_ids[i]), float(pos[i, 0]),
                                      float(pos[i, 1]), 0.0) for i in covered])
            lo = np.searchsorted(s.t, t, "left"); hi = np.searchsorted(s.t, t + KEYFRAME_DT_S, "left")
            if hi > lo:
                dx = s.x[lo:hi]; dy = s.y[lo:hi]; th = s.theta[lo:hi]; rh = s.rho[lo:hi]; tt = s.t[lo:hi]
                m = ((dx - pose[0]) ** 2 + (dy - pose[1]) ** 2) <= r2
                for i in np.nonzero(m)[0]:
                    model.add_detection(int(tt[i] * 1e9), float(dx[i]), float(dy[i]),
                                        float(th[i]), float(rh[i]), float(pose[0]), float(pose[1]))
                    pushed += 1
            if k % 40000 == 0 and k:
                print(f"    kf {k}/{n_kf} pushed={pushed} ({k/(time.time()-tic):.0f} kf/s)")
        print(f"[rheos_atc]   {pathlib.Path(day).stem}: pushed {pushed} in {time.time()-tic:.0f}s")

    print(f"[rheos_atc] covered {len(covered)} places; fitting ...")
    model.fit()
    places = list(model.export_places())

    # --- online scoring: stream held-out detections, accumulate, no preds file ---
    flow = FlowAccumulator()
    n = 0; tic = time.time()
    for day in args.test_days:
        s = AtcStream(day, x_bounds=xb, y_bounds=yb)
        for lo in range(0, len(s.t), args.chunk):
            hi = min(lo + args.chunk, len(s.t))
            matched = []; lpj = []; lph = []; lps = []; mr = []
            for i in range(lo, hi):
                x = float(s.x[i]); y = float(s.y[i]); th = float(s.theta[i]); rh = float(s.rho[i])
                lp = model.log_prob(x, y, th, rh)
                ok = not math.isnan(lp)
                matched.append(ok)
                lpj.append(lp)
                lph.append(model.log_prob_heading(x, y, th) if ok else math.nan)
                lps.append(model.log_prob_speed(x, y, rh) if ok else math.nan)
                mr.append(_predict_rho(places, x, y, cfg.assoc_radius) if ok else math.nan)
            flow.update(np.array(matched, bool), s.rho[lo:hi],
                        {"heading": lph, "speed": lps, "joint": lpj}, mr)
            n = flow.n_total
            print(f"    scored {n} ({n/(time.time()-tic):.0f}/s)")
    f = flow.result()
    lines = [
        f"n={f['n_total']} match_rate={f['match_rate']:.4f} rho_max={f['rho_max']:.3f}",
        f"MLPD heading : {f['mlpd_heading']:.4f}",
        f"MLPD speed   : {f['mlpd_speed']:.4f}",
        f"MLPD joint   : {f['mlpd_joint']:.4f}",
        f"Speed MAE    : {f['speed_mae_mps']:.4f}",
    ]
    print("[rheos_atc] FLOW\n  " + "\n  ".join(lines))
    (out / "metrics_online.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
