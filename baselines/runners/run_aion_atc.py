#!/usr/bin/env python3
r"""Aion baseline on the ATC simulation (train + online-score, no preds files).

Aion is the online scene-graph spectral baseline (per-node orientation histograms
+ FreMEn); on ATC it is the scalable stand-in for the batch STeF-Map (same FreMEn
prediction, fit incrementally on a scene graph). No reconstruction frontend: the
simulated robot patrol supplies the node layer and its pose, exactly as
train_atc.py does for Kairos, but detections go to AionCore.update_node_positions
/ add_detection / flush_window. After training, the model is scored over the
held-out day(s) and the flow metrics accumulate in one streaming pass through the
SHARED online_metrics accumulators (same math evaluate.py uses); nothing
per-detection is written. Aion is heading-only, so only the heading MLPD (and
coverage) are meaningful; speed/joint are reported as "--".

FreMEn windows reset per session (per day) as in aion's own runner, while the
absolute window times fed to flush_window preserve the daily/weekly phase across
sessions. Runs in the aion Docker image.

Usage:
    python3 run_aion_atc.py --graph <atc_graph.json> --days <train npz...> \
        --test-days <sep npz...> --out-dir results/atc_aion \
        --candidate-periods 3600 43200 86400 604800
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

from aion_python import AionCore, AionConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--days", nargs="*", default=[], help="train-day npz/csv (may be empty with --load-state)")
    ap.add_argument("--test-days", nargs="+", required=True, help="held-out test-day npz/csv")
    ap.add_argument("--save-state", default=None,
                    help="persist the trained model here after the training pass")
    ap.add_argument("--load-state", default=None,
                    help="restore a trained model and skip training entirely")
    ap.add_argument("--dump-rows", action="store_true",
                    help="write per-day rows_<day>.npz (matched, lp, masses) for "
                         "paired / reweighted analysis")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--assoc-radius", type=float, default=0.7)
    ap.add_argument("--grid-size", type=float, default=0.4)
    ap.add_argument("--time-window", type=float, default=10.0, dest="time_window_s")
    ap.add_argument("--order", type=int, default=1, dest="fremen_order")
    ap.add_argument("--candidate-periods", type=float, nargs="+",
                    default=[3600.0, 43200.0, 86400.0, 604800.0])
    ap.add_argument("--map-pgm", default=None,
                    help="ATC occupancy PGM; with --map-yaml, use the map extent "
                         "as the detection rectangle (the canonical scoring row set) "
                         "instead of the nav-graph box")
    ap.add_argument("--map-yaml", default=None)
    ap.add_argument("--patrol-policy", default="serpentine",
                    choices=["serpentine", "random"],
                    help="training patrol: the fixed serpentine loop, or "
                         "uniformly random regions (the sparse-coverage study)")
    ap.add_argument("--policy-seed", type=int, default=7)
    ap.add_argument("--policy-regions", type=int, default=24)
    ap.add_argument("--explore-ratio", type=float, default=1.0,
                    help="fraction of the day's 10-min decision intervals the "
                         "robot observes (Molina et al.'s exploration ratio); "
                         "active intervals spread by even stride; 1.0 = all day")
    ap.add_argument("--robot-speed", type=float, default=ROBOT_SPEED_MPS,
                    help="robot travel speed (m/s)")
    ap.add_argument("--fov-m", type=float, default=ACTIVE_WINDOW_RADIUS_M,
                    help="active-window (field-of-view) radius for the robot")
    ap.add_argument("--stream-source", default="atc", choices=["atc", "hb"],
                    help="'hb': --days/--test-days are ISO dates of a "
                         "station stream, interpolated per track to --interp-hz")
    ap.add_argument("--hb-root", default="/data/hb-station/Data-2")
    ap.add_argument("--station", choices=["HB", "PB"], default="HB")
    ap.add_argument("--interp-hz", type=float, default=10.0)
    ap.add_argument("--gap-max-s", type=float, default=2.0)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cfg = AionConfig()
    cfg.assoc_radius = args.assoc_radius
    cfg.grid_size = args.grid_size
    cfg.fremen_order = args.fremen_order
    cfg.candidate_periods = list(args.candidate_periods)
    model = AionCore(cfg)
    win_ns = int(args.time_window_s * 1e9)

    pos, node_ids, adj, trav = load_nav_graph(args.graph)
    poly, _ = build_coverage_loop(pos, adj, trav)
    sim = RobotSim(poly, pos, radius=args.fov_m)
    if args.map_pgm and args.map_yaml:
        # canonical rectangle: the dataset's occupancy-map extent (same parse
        # as run_stefmap_atc._bounds, so every scorer sees the same row set)
        from PIL import Image
        res = origin = None
        with open(args.map_yaml) as f:
            for line in f:
                if line.startswith("resolution:"):
                    res = float(line.split(":")[1])
                if line.startswith("origin:"):
                    origin = [float(v) for v in line.split("[")[1].split("]")[0].split(",")]
        img = np.array(Image.open(args.map_pgm))
        H, W = img.shape[:2]
        xb = (origin[0], origin[0] + W * res)
        yb = (origin[1], origin[1] + H * res)
    else:
        xb = (pos[:, 0].min() - 6, pos[:, 0].max() + 6)
        yb = (pos[:, 1].min() - 6, pos[:, 1].max() + 6)
    r2 = args.fov_m ** 2
    print(f"[aion_atc] {len(pos)} nodes, coverage loop {sim.total:.0f} m, "
          f"{len(args.days)} train days, periods {cfg.candidate_periods}")

    stream_open = None
    if args.stream_source == "hb":
        from hb_stream import HBDayStream, hb_day_files

        def stream_open(day):
            files = hb_day_files(args.hb_root, args.station, day)
            if not files:
                raise SystemExit(f"[aion_atc] no hb files for {day}")
            return HBDayStream(files, x_bounds=xb, y_bounds=yb,
                                  interp_hz=args.interp_hz,
                                  gap_max_s=args.gap_max_s)

    covered = set()
    def nodes_of(cov):
        return [(int(node_ids[i]), float(pos[i, 0]), float(pos[i, 1]), 0.0) for i in cov]

    if args.load_state:
        ok, err = model.load_state(args.load_state)
        if not ok:
            raise SystemExit(f"[aion_atc] load_state failed: {err}")
        print(f"[aion_atc] loaded state from {args.load_state} "
              f"(node entries {model.num_node_entries}, hash cells {model.num_hash_cells})")
        args.days = []

    if args.patrol_policy != "serpentine":
        from explore_patrol import (DECIDE_DT_S, RegionPatrols, active_intervals,
                                    day_polyline, make_regions, sample_schedule)
        labels, _ = make_regions(pos, trav, k=args.policy_regions,
                                 seed=args.policy_seed)
        patrols = RegionPatrols(pos, adj, trav, labels, args.policy_regions)
        prng = np.random.default_rng(args.policy_seed)
        print(f"[aion_atc] policy '{args.patrol_policy}': "
              f"{args.policy_regions} regions")

    for di, day in enumerate(args.days):
        s = (AtcStream(day, x_bounds=xb, y_bounds=yb) if stream_open is None
             else stream_open(day))
        t0 = s.t_min
        dur = (s.t_max - t0) if args.minutes == 0 else args.minutes * 60.0
        if args.patrol_policy != "serpentine":
            n_iv = max(1, int(np.ceil(dur / DECIDE_DT_S)))
            w = np.ones(args.policy_regions)
            schedule = sample_schedule(w, n_iv, prng)
            sim = RobotSim(day_polyline(patrols, schedule, dur,
                                        speed=args.robot_speed), pos,
                           radius=args.fov_m)
            active_mask = active_intervals(n_iv, args.explore_ratio)
            print(f"[aion_atc]   policy day {di+1}: {n_iv} intervals "
                  f"({int(active_mask.sum())} observed, "
                  f"ratio {args.explore_ratio:g}), loop {sim.total:.0f} m")
        else:
            active_mask = None
        n_kf = int(dur / KEYFRAME_DT_S)
        next_win_end = None
        tic = time.time(); pushed = 0
        for k in range(n_kf):
            t = t0 + k * KEYFRAME_DT_S
            if active_mask is not None:
                iv = int((k * KEYFRAME_DT_S) / DECIDE_DT_S)
                if iv >= len(active_mask) or not active_mask[iv]:
                    continue  # robot occupied elsewhere this interval
            pose = sim.pose_at_arclength(args.robot_speed * k * KEYFRAME_DT_S)
            vis = sim.visible_places(pose)
            new = [i for i in vis if i not in covered]
            if new:
                covered.update(new)
                model.update_node_positions(nodes_of(covered))
            lo = np.searchsorted(s.t, t, "left"); hi = np.searchsorted(s.t, t + KEYFRAME_DT_S, "left")
            if hi > lo:
                dx = s.x[lo:hi]; dy = s.y[lo:hi]; th = s.theta[lo:hi]; tt = s.t[lo:hi]
                m = ((dx - pose[0]) ** 2 + (dy - pose[1]) ** 2) <= r2
                for i in np.nonzero(m)[0]:
                    t_ns = int(tt[i] * 1e9)
                    if next_win_end is None:
                        next_win_end = t_ns + win_ns
                    while t_ns >= next_win_end:
                        model.flush_window(next_win_end); next_win_end += win_ns
                    model.add_detection(t_ns, float(dx[i]), float(dy[i]),
                                        float(th[i]), float(pose[0]), float(pose[1]))
                    pushed += 1
            if k % 40000 == 0 and k:
                print(f"    kf {k}/{n_kf} pushed={pushed} ({k/(time.time()-tic):.0f} kf/s)")
        model.update_node_positions(nodes_of(covered))
        if next_win_end is not None:
            model.flush_window(next_win_end)
        print(f"[aion_atc]   {pathlib.Path(day).stem}: pushed {pushed} in {time.time()-tic:.0f}s")

    if args.save_state:
        ok, err = model.save_state(args.save_state)
        if not ok:
            raise SystemExit(f"[aion_atc] save_state failed: {err}")
        print(f"[aion_atc] saved state to {args.save_state}")

    print(f"[aion_atc] covered {len(covered)} nodes; scoring ...")
    # online scoring: heading-only. One heading_distribution call per detection
    # yields the full 8-bin mass vector; the observed heading's log-density is
    # log(mass[theta_to_bin(theta)] * 8/(2pi)), the same value log_prob_heading
    # returns (asserted against it on the first rows of every run).
    from online_metrics import HistogramHeadingAccumulator, CellMeanAccumulator
    DENS = 8.0 / (2.0 * math.pi)
    HEADING_FLOOR = -math.log(2.0 * math.pi)
    PARITY_ROWS = 200_000
    hd = HistogramHeadingAccumulator(n_bins=8)
    flow = FlowAccumulator()
    cellmean = CellMeanAccumulator(voxel_size=args.grid_size)
    n_matched_total = 0
    n_parity = 0
    tic = time.time()
    for day in args.test_days:
        s = (AtcStream(day, x_bounds=xb, y_bounds=yb) if stream_open is None
             else stream_open(day))
        day_matched = []; day_lp = []; day_masses = []
        for lo in range(0, len(s.t), args.chunk):
            hi = min(lo + args.chunk, len(s.t))
            nchunk = hi - lo
            matched = np.zeros(nchunk, bool)
            lph = np.full(nchunk, math.nan)
            masses = np.zeros((nchunk, 8))
            for i in range(lo, hi):
                t_q = int(s.t[i] * 1e9); x_q = float(s.x[i]); y_q = float(s.y[i])
                d = model.heading_distribution(t_q, x_q, y_q)
                if d:
                    j = i - lo
                    matched[j] = True
                    masses[j] = d
                    lph[j] = math.log(d[model.theta_to_bin(float(s.theta[i]))] * DENS)
                    if n_parity < PARITY_ROWS:
                        ref = model.log_prob_heading(t_q, x_q, y_q, float(s.theta[i]))
                        if not abs(ref - lph[j]) < 1e-9:
                            raise SystemExit(f"[aion_atc] PARITY FAIL at row {i}: "
                                             f"{ref} vs {lph[j]}")
                        n_parity += 1
            n_matched_total += int(matched.sum())
            crps_r, mpp_r, good = hd.rows(matched, s.theta[lo:hi], masses)
            hd.accumulate(crps_r, mpp_r, good)
            lp_full = np.where(good & np.isfinite(lph), lph, HEADING_FLOOR)
            xyz = np.column_stack([s.x[lo:hi], s.y[lo:hi], np.zeros(nchunk)])
            cellmean.update(xyz, good, lp_full, crps_r, mpp_r)
            flow.update(matched, s.rho[lo:hi],
                        {"heading": lph, "speed": [math.nan] * nchunk, "joint": [math.nan] * nchunk},
                        [math.nan] * nchunk)
            if args.dump_rows:
                day_matched.append(matched); day_lp.append(lph.astype(np.float32))
                day_masses.append(masses.astype(np.float32))
            print(f"    scored {flow.n_total} ({flow.n_total/(time.time()-tic):.0f}/s)")
        if args.dump_rows:
            stem = pathlib.Path(day).stem
            np.savez_compressed(out / f"rows_{stem}.npz",
                                matched=np.concatenate(day_matched),
                                lp=np.concatenate(day_lp),
                                masses=np.concatenate(day_masses))
            print(f"    wrote rows_{stem}.npz")
    if n_parity < min(PARITY_ROWS, n_matched_total):
        raise SystemExit("[aion_atc] parity gate never engaged; refusing to report")
    print(f"[aion_atc] parity gate: {n_parity} rows checked against log_prob_heading")
    f = flow.result()
    h = hd.result()
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"
    lines = [
        f"n={f['n_total']} match_rate={f['match_rate']:.4f}",
        f"MLPD heading : {fmt(f['mlpd_heading'])}",
        f"MLPD speed   : {fmt(f['mlpd_speed'])}   (heading-only model)",
        f"MLPD joint   : {fmt(f['mlpd_joint'])}   (heading-only model)",
        f"circular CRPS: {h['crps_rad']:.4f}  [matched-only {h['crps_rad_matched_only']:.4f}]  (uniform 0.7854)",
        f"MPP          : {h['mpp']:.4f}  [matched-only {h['mpp_matched_only']:.4f}]  (uniform 0.1250)",
    ]
    print("[aion_atc] FLOW\n  " + "\n  ".join(lines))
    (out / "metrics_online.txt").write_text("\n".join(lines) + "\n")

    # Per-cell heading metric (equal weight per cell; exploration aggregation).
    cm = cellmean.result()
    if cm is not None:
        lo_ci, hi_ci = cm["crps_ci95"]
        cell_lines = [
            f"n_cells={cm['n_cells']} median_det_per_cell={cm['median_det_per_cell']:.0f}",
            f"MLPD heading : {cm['mlpd_heading']:.4f}   (per-cell equal weight)",
            f"circular CRPS: {cm['crps_rad']:.4f}   [bootstrap 95% CI over cells "
            f"{lo_ci:.4f}, {hi_ci:.4f}]  (uniform floor 0.7854)",
            f"MPP          : {cm['mpp']:.4f}   (uniform 0.1250)",
            "by min-detections/cell:",
        ] + [f"  >= {t:>3}: {v['n_cells']} cells  CRPS {v['crps_rad']:.4f}  "
             f"MLPD {v['mlpd_heading']:.4f}" for t, v in sorted(cm["by_threshold"].items())]
        print("[aion_atc] PER-CELL\n  " + "\n  ".join(cell_lines))
        (out / "metrics_percell.txt").write_text("\n".join(cell_lines) + "\n")
        np.savez(out / "cellmap.npz", vx=cm["cell_vx"], vy=cm["cell_vy"],
                 crps=cm["per_cell_crps"], mlpd=cm["per_cell_mlpd"],
                 mpp=cm["per_cell_mpp"], n=cm["per_cell_n"],
                 voxel_size=np.float64(args.grid_size))
        print(f"[aion_atc] per-cell heatmap arrays -> {out / 'cellmap.npz'}")

    import json
    (out / "metrics_online.json").write_text(json.dumps({
        "test_days": [pathlib.Path(d).stem for d in args.test_days],
        "n": f["n_total"], "n_matched": n_matched_total,
        "match_rate": f["match_rate"],
        "mlpd_heading": f["mlpd_heading"],
        "mlpd_heading_matched_only": f["mlpd_heading_matched_only"],
        "crps": h["crps_rad"], "crps_matched_only": h["crps_rad_matched_only"],
        "mpp": h["mpp"], "mpp_matched_only": h["mpp_matched_only"],
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
