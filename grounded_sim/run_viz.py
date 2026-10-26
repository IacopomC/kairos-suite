#!/usr/bin/env python3
r"""Dataset-agnostic Rerun recording driver for Kairos grounded simulations.

Emits the portable recording (``.npz``) that ``kairos_visualizer`` renders to a
scrubbable ``.rrd``. Two modes:

  online   drive the embodied robot across ``--days`` and snapshot the ONLINE
           model as it grows -- coverage fills in, per-cell counts rise, and the
           per-node flow arrows sharpen (the map-building view).
  predict  load a FROZEN ``--flow-state`` and query its forecast across
           ``--days``, overlaying the real detections of those days -- prediction
           vs. reality on data the model never trained on.

The stream source is selected exactly as in ``train_atc.py``
(``--stream-source {atc,hb}``): a "day" is an npz/csv path (atc) or an ISO
date (hb). Everything else -- the coverage loop, the field-of-view gating, the
snapshot, the detection layer, and the recording schema -- is dataset-agnostic;
bounds come from the authored graph. To add a dataset, add a stream class with
the ``AtcStream`` interface and a branch in ``make_stream_opener``.

Runs in the pipeline container (Python 3.8); rendering the ``.npz`` needs none of
this (see ``kairos_visualizer``).

Usage:
    # watch the model being built by the roaming robot
    python3 run_viz.py --graph /data/atc/graph/atc_graph.json \
        --days /data/atc/npz/atc-20121114.npz --scene-dir stub_scene \
        --mode online --out-npz out.npz

    # forecast from a frozen state over a held-out day, real detections overlaid
    python3 run_viz.py --graph ... --days <test npz> --scene-dir stub_scene \
        --mode predict --flow-state OUT/train/flow_state.bin --out-npz out.npz

Writes a portable .npz. Render it to a Rerun .rrd with kairos_visualizer, or
let create_rrd.sh do both steps.
"""
import argparse
import pathlib
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scene_loader import get_dataloader
from kairos_python import flow_archetype
from kairos_python._kairos_bindings import classify_archetype
from kairos_python.pipeline import (
    load_pipeline, load_configs, archetype_params_from_config,
    archetype_horizon_from_config)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from robot_sim import (load_nav_graph, build_coverage_loop, RobotSim,
                       KEYFRAME_DT_S, ROBOT_SPEED_MPS, ACTIVE_WINDOW_RADIUS_M)
from atc_stream import AtcStream
from flow_viz import load_nav_layer

VOXEL_SIZE = 0.4
MAXC = 3        # top mixture components kept per node for the arrows


def make_stream_opener(args, xb, yb):
    """Return ``open(day) -> stream`` for the chosen source; the stream exposes
    the AtcStream interface (.t/.x/.y/.theta/.rho/.id/.t_min/.t_max/.between)."""
    if args.stream_source == "hb":
        from hb_stream import HBDayStream, hb_day_files

        def opener(day):
            files = hb_day_files(args.hb_root, args.station, day)
            if not files:
                raise SystemExit(f"[run_viz] no hb files for {day}")
            return HBDayStream(files, x_bounds=xb, y_bounds=yb,
                                  interp_hz=args.interp_hz, gap_max_s=args.gap_max_s)
        return opener

    def opener(day):
        return AtcStream(day, x_bounds=xb, y_bounds=yb)
    return opener


def read_place_summaries(pipeline, tq, cov, N, arch_params, arch_horizon_s):
    """Forecast at time ``tq`` for the covered nodes ``cov`` -> per-node arrays.

    ``arch_horizon_s`` is the classification horizon the archetype config names:
    a place stores an occupancy, so the probability the STATIC gate compares
    against ``min_probability`` only has a meaning once a horizon is supplied.
    """
    pipeline.update_place_summaries(float(tq))
    ps = pipeline.place_summaries(arch_horizon_s)
    sup = np.asarray(ps["support"]); prob = np.asarray(ps["probability"])
    dt = np.asarray(ps["dominant_theta"]); dr = np.asarray(ps["dominant_rho"])
    dth = np.full(N, np.nan); drho = np.zeros(N)
    act = np.zeros(N, bool); arch = np.zeros(N, np.int8)
    cth = np.zeros((N, MAXC)); cwt = np.zeros((N, MAXC))
    for a, i in enumerate(cov):
        if sup[a] <= 0:
            continue
        w = np.asarray(ps["comp_weight"][a], float)
        th = np.asarray(ps["comp_theta"][a], float)
        act[i] = True; dth[i] = float(dt[a]); drho[i] = float(dr[a])
        arch[i] = classify_archetype(float(prob[a]), int(sup[a]), w.tolist(), **arch_params)
        order = np.argsort(-w)[:MAXC]
        cth[i, :len(order)] = th[order]; cwt[i, :len(order)] = w[order]
    return dth, drho, act, arch, cth, cwt


def build_day_tracks(stream, t0, dur, base_step, voxtree, det_viz_dt, decimate,
                     robot_pose_fn=None):
    """Per-person interpolated detection arrows for one day on a local playback
    grid; ``step`` is offset by ``base_step`` to index the concatenated global
    ``det_times``. Decimation keeps whole tracks (1 person in ``decimate``), so
    each kept person moves continuously rather than flickering. ``robot_pose_fn``
    (t -> xy) flags in-window arrows; absent (no robot) -> all False."""
    T = np.arange(t0, t0 + dur, det_viz_dt)
    tracks = pd.DataFrame({"id": stream.id, "t": stream.t, "x": stream.x,
                           "y": stream.y, "th": stream.theta})
    S_, X_, Y_, TH_ = [], [], [], []
    for pid_val, g in tracks.groupby("id", sort=False):
        if decimate > 1 and int(pid_val) % decimate != 0:
            continue
        tt = g.t.values
        if len(tt) < 2:
            continue
        i0 = np.searchsorted(T, tt[0], "left"); i1 = np.searchsorted(T, tt[-1], "right")
        if i1 <= i0:
            continue
        Ts = T[i0:i1]
        px = np.interp(Ts, tt, g.x.values); py = np.interp(Ts, tt, g.y.values)
        j = np.clip(np.searchsorted(tt, Ts), 1, len(tt) - 1)
        jn = np.where(np.abs(tt[j] - Ts) < np.abs(tt[j - 1] - Ts), j, j - 1)
        S_.append(np.arange(i0, i1)); X_.append(px); Y_.append(py)
        TH_.append(g.th.values[jn])
    if not S_:
        z = np.zeros(0)
        return T, np.zeros(0, np.int32), z, z, z, np.zeros(0, bool), np.zeros(0, np.int32)
    step = np.concatenate(S_).astype(np.int64)
    x = np.concatenate(X_); y = np.concatenate(Y_); th = np.concatenate(TH_)
    if robot_pose_fn is not None:
        rr = np.array([robot_pose_fn(t0 + s * det_viz_dt) for s in step])
        din = ((x - rr[:, 0]) ** 2 + (y - rr[:, 1]) ** 2) <= ACTIVE_WINDOW_RADIUS_M ** 2
    else:
        din = np.zeros(len(step), bool)
    dvd, dvi = voxtree.query(np.column_stack([x, y]))
    dvox = np.where(dvd <= VOXEL_SIZE, dvi, -1).astype(np.int32)
    o = np.argsort(step, kind="stable")
    return (T, (step[o] + base_step).astype(np.int32), x[o], y[o], th[o],
            din[o], dvox[o])


def run_online(args, pipeline, arch_params, arch_horizon_s, pos, nav_r, nav_edges,
               sim, opener):
    N = len(pos)
    vox_xyz = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)]
    nav_r_l = [float(nav_r[i]) for i in range(N)]
    voxtree = cKDTree(pos)
    r2 = args.fov_m ** 2

    vox_count = np.zeros(N); vox_create_t = np.full(N, -1.0)
    vox_active_t = np.full(N, -1.0)
    # prebuilt map: the whole nav layer is known from t0 (static geometry); otherwise
    # cells appear as the robot's coverage reaches them.
    cover_t = np.zeros(N) if args.prebuilt_map else np.full(N, -1.0)
    snap_t, sd_th, sd_rho, sd_act, sd_arch, sd_cth, sd_cwt = [], [], [], [], [], [], []
    robot_t_all, robot_xy_all = [], []
    dt_all, ds_all, dx_all, dy_all, dth_all, din_all, dvox_all = [], [], [], [], [], [], []
    step_offset = 0

    def snapshot(t):
        pipeline.flush_detections()
        cov = np.where(cover_t >= 0)[0]
        if len(cov):
            # Edges among the covered nodes, re-indexed to the covered list:
            # evidence sharing reads the layer's edges for its neighbourhoods.
            sub = {int(g): li for li, g in enumerate(cov)}
            cov_edges = [[sub[int(i)], sub[int(j)]] for i, j in nav_edges
                         if int(i) in sub and int(j) in sub]
            pipeline.set_navigational_layer([vox_xyz[i] for i in cov],
                                            [nav_r_l[i] for i in cov],
                                            cov_edges)
            dth, drho, act, arch, cth, cwt = read_place_summaries(
                pipeline, t, cov, N, arch_params, arch_horizon_s)
        else:
            dth = np.full(N, np.nan); drho = np.zeros(N); act = np.zeros(N, bool)
            arch = np.zeros(N, np.int8); cth = np.zeros((N, MAXC)); cwt = np.zeros((N, MAXC))
        snap_t.append(float(t))
        sd_th.append(dth); sd_rho.append(drho); sd_act.append(act)
        sd_arch.append(arch); sd_cth.append(cth); sd_cwt.append(cwt)
        print(f"  snapshot t={t:.0f}: covered={len(cov)} active={int(act.sum())}", flush=True)

    run_start = None
    for di, day in enumerate(args.days):
        stream = opener(day)
        t0 = stream.t_min
        if run_start is None:
            run_start = t0
        dur = (stream.t_max - t0) if args.minutes == 0 else args.minutes * 60.0
        n_kf = int(dur / KEYFRAME_DT_S)
        next_snap = t0
        pushed = 0; tic = time.time()
        print(f"[run_viz] online day {di+1}/{len(args.days)} {day}: {n_kf} keyframes "
              f"({dur/3600:.1f} h)", flush=True)
        for k in range(n_kf):
            t = t0 + k * KEYFRAME_DT_S
            pose = sim.pose_at_arclength(ROBOT_SPEED_MPS * k * KEYFRAME_DT_S)
            vis = sim.visible_places(pose)
            if vis:
                va = np.asarray(vis, int)
                cover_t[va[cover_t[va] < 0]] = t
            pipeline.set_visible_voxels(
                VOXEL_SIZE, [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in vis])
            dx, dy, th, rh, pid = stream.between(t, t + KEYFRAME_DT_S)
            if len(dx):
                m = ((dx - pose[0]) ** 2 + (dy - pose[1]) ** 2) <= r2
                dx, dy, th, rh, pid = dx[m], dy[m], th[m], rh[m], pid[m]
            pipeline.push_detections(
                t, [[float(x), float(y), 0.0] for x, y in zip(dx, dy)],
                [float(v) for v in th], [float(v) for v in rh],
                track_ids=[int(v) for v in pid])
            pipeline.flush_detections()
            pushed += len(dx)
            if len(dx):
                dvd, dvi = voxtree.query(np.column_stack([dx, dy]))
                hit = dvd <= VOXEL_SIZE
                np.add.at(vox_count, dvi[hit], 1)
                uh = np.unique(dvi[hit])
                vox_create_t[uh[vox_create_t[uh] < 0]] = t
                na = uh[(vox_active_t[uh] < 0) & (vox_count[uh] >= args.active_n)]
                vox_active_t[na] = t
            if t >= next_snap:
                snapshot(t)
                fine = (args.fine_snapshot_s > 0 and
                        (t - run_start) < args.fine_until_h * 3600.0)
                next_snap += args.fine_snapshot_s if fine else args.snapshot_every_s
        snapshot(t0 + dur)
        rt = np.arange(t0, t0 + dur, args.robot_dt)
        robot_t_all.append(rt)
        robot_xy_all.append(np.array([sim.pose_at_arclength(ROBOT_SPEED_MPS * (r - t0))
                                      for r in rt]))
        pose_fn = lambda tt, _t0=t0: sim.pose_at_arclength(ROBOT_SPEED_MPS * (tt - _t0))
        dT, ds, dxk, dyk, dthk, dink, dvk = build_day_tracks(
            stream, t0, dur, step_offset, voxtree, args.det_viz_dt,
            args.det_decimate, robot_pose_fn=pose_fn)
        dt_all.append(dT); ds_all.append(ds); dx_all.append(dxk); dy_all.append(dyk)
        dth_all.append(dthk); din_all.append(dink); dvox_all.append(dvk)
        step_offset += len(dT)
        print(f"[run_viz]   done {day}: {pushed} det in {time.time()-tic:.0f}s", flush=True)

    session_t0 = float(np.concatenate(robot_t_all)[0])
    if args.prebuilt_map:
        cover_t[:] = session_t0          # absolute, as for the coverage-driven case
    return dict(
        t0=session_t0, t1=float(np.concatenate(robot_t_all)[-1]),
        nav_pos=pos, nav_edges=nav_edges, nav_radius=nav_r, nav_appear_t=cover_t,
        room_xy=np.array([pos[:, 0].mean(), pos[:, 1].mean()]),
        vox_xy=pos, vox_count=vox_count,
        vox_create_t=vox_create_t, vox_active_t=vox_active_t,
        flow_times=np.array(snap_t),
        node_dtheta=np.array(sd_th), node_drho=np.array(sd_rho),
        node_active=np.array(sd_act), node_arch=np.array(sd_arch),
        node_ctheta=np.array(sd_cth), node_cweight=np.array(sd_cwt),
        det_times=np.concatenate(dt_all), det_step=np.concatenate(ds_all),
        det_x=np.concatenate(dx_all), det_y=np.concatenate(dy_all),
        det_th=np.concatenate(dth_all), det_in=np.concatenate(din_all),
        det_vox=np.concatenate(dvox_all),
        robot_times=np.concatenate(robot_t_all), robot_xy=np.concatenate(robot_xy_all),
        arch_colors=np.array(flow_archetype.color_list()),
        active_window_radius=args.fov_m)


def run_predict(args, pipeline, arch_params, arch_horizon_s, pos, nav_r, nav_edges,
                opener):
    N = len(pos)
    voxtree = cKDTree(pos)
    # whole mapped graph is known for a fixed frozen model: set the nav layer once.
    pipeline.set_navigational_layer(
        [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)],
        [float(nav_r[i]) for i in range(N)],
        [[int(i), int(j)] for i, j in nav_edges])
    cov = np.arange(N)

    vox_count = np.zeros(N)
    snap_t, sd_th, sd_rho, sd_act, sd_arch, sd_cth, sd_cwt = [], [], [], [], [], [], []
    dt_all, ds_all, dx_all, dy_all, dth_all, dvox_all = [], [], [], [], [], []
    step_offset = 0

    for di, day in enumerate(args.days):
        stream = opener(day)
        t0, t1 = stream.t_min, stream.t_max
        tic = time.time()
        qs = np.arange(t0, t1, args.query_every_s)
        for tq in qs:
            dth, drho, act, arch, cth, cwt = read_place_summaries(
                pipeline, tq, cov, N, arch_params, arch_horizon_s)
            snap_t.append(float(tq))
            sd_th.append(dth); sd_rho.append(drho); sd_act.append(act)
            sd_arch.append(arch); sd_cth.append(cth); sd_cwt.append(cwt)
        dvd, dvi = voxtree.query(np.column_stack([stream.x, stream.y]))
        np.add.at(vox_count, dvi[dvd <= VOXEL_SIZE], 1)
        dT, ds, dxk, dyk, dthk, _din, dvk = build_day_tracks(
            stream, t0, t1 - t0, step_offset, voxtree, args.det_viz_dt,
            args.det_decimate, robot_pose_fn=None)
        dt_all.append(dT); ds_all.append(ds); dx_all.append(dxk); dy_all.append(dyk)
        dth_all.append(dthk); dvox_all.append(dvk)
        step_offset += len(dT)
        print(f"[run_viz] predict {day} ({di+1}/{len(args.days)}): "
              f"{len(qs)} queries, {time.time()-tic:.0f}s", flush=True)

    det_step = np.concatenate(ds_all)
    return dict(
        t0=float(snap_t[0]), t1=float(snap_t[-1]),
        nav_pos=pos, nav_edges=nav_edges, nav_radius=nav_r,
        # A frozen state is known in full from the first snapshot on. Appearance
        # times are absolute (see schema.py), so they carry that instant rather
        # than zero, which would place the whole substrate an epoch before the
        # session and stretch the timeline to the point of being unusable.
        nav_appear_t=np.full(N, float(snap_t[0])),
        room_xy=np.array([pos[:, 0].mean(), pos[:, 1].mean()]),
        vox_xy=pos, vox_count=vox_count,
        vox_create_t=np.full(N, float(snap_t[0])),
        vox_active_t=np.full(N, float(snap_t[0])),
        flow_times=np.array(snap_t),
        node_dtheta=np.array(sd_th), node_drho=np.array(sd_rho),
        node_active=np.array(sd_act), node_arch=np.array(sd_arch),
        node_ctheta=np.array(sd_cth), node_cweight=np.array(sd_cwt),
        det_times=np.concatenate(dt_all), det_step=det_step,
        det_x=np.concatenate(dx_all), det_y=np.concatenate(dy_all),
        det_th=np.concatenate(dth_all), det_in=np.zeros(len(det_step), bool),
        det_vox=np.concatenate(dvox_all),
        # no robot on a fixed-camera test set; omit it so the renderer skips the
        # layer entirely (parking it off-map would blow up the view bounds).
        arch_colors=np.array(flow_archetype.color_list()),
        active_window_radius=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--days", nargs="+", required=True,
                    help="npz/csv paths (atc) or ISO dates (hb)")
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--config-name", default="atc")
    ap.add_argument("--labelspace", default="ade20k_outdoor")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--mode", choices=["online", "predict"], default="online")
    ap.add_argument("--flow-state", default=None, help="frozen state (--mode predict)")
    ap.add_argument("--stream-source", default="atc", choices=["atc", "hb"])
    ap.add_argument("--hb-root", default="/data/hb-station/Data-2")
    ap.add_argument("--station", choices=["HB", "PB"], default="HB")
    ap.add_argument("--interp-hz", type=float, default=10.0)
    ap.add_argument("--gap-max-s", type=float, default=2.0)
    ap.add_argument("--fov-m", type=float, default=ACTIVE_WINDOW_RADIUS_M)
    ap.add_argument("--minutes", type=float, default=0.0, help="0 = whole day")
    ap.add_argument("--active-n", type=int, default=10)
    ap.add_argument("--robot-dt", type=float, default=0.5)
    ap.add_argument("--det-viz-dt", type=float, default=1.0)
    ap.add_argument("--det-decimate", type=int, default=1,
                    help="keep 1 person in N (whole tracks, so motion stays smooth)")
    ap.add_argument("--snapshot-every-s", type=float, default=1200.0,
                    help="online: coarse snapshot cadence")
    ap.add_argument("--fine-snapshot-s", type=float, default=0.0,
                    help="online: finer cadence for the first --fine-until-h (0=off)")
    ap.add_argument("--fine-until-h", type=float, default=0.0)
    ap.add_argument("--query-every-s", type=float, default=120.0,
                    help="predict: forecast-query cadence within each day")
    ap.add_argument("--arch-horizon-s", type=float, default=60.0,
                    help="presence horizon (s) for the archetype gate: a node is "
                         "Static if its presence over this interval is below "
                         "min_probability. Matches a presence-estimation horizon; "
                         "1 s (instantaneous) leaves nearly every cell Static.")
    ap.add_argument("--occupancy", default=None,
                    help="authored occupancy-layer .npz (free_xy/wall_xy from "
                         "author_graph.py --out-occupancy); adds the occupancy-map "
                         "wall cells to the recording so the renderer draws them.")
    ap.add_argument("--prebuilt-map", action="store_true",
                    help="online: treat the whole map as already built (nav layer "
                         "known from t0, no incremental coverage) so the geometry is "
                         "static and only the flow evolves. Use with --flow-state for "
                         "a window whose 3DSG already exists and whose model is "
                         "inherited from earlier days (e.g. a mid/late training week).")
    args = ap.parse_args()

    if args.mode == "predict" and not args.flow_state:
        raise SystemExit("[run_viz] --mode predict requires --flow-state")

    data = get_dataloader(args.scene_dir)
    pipeline = load_pipeline(data, args.config_name, args.labelspace, enable_lcd=False,
                             config_overrides={"flow_temporal":
                                               {"place_summary_max_nodes": 0,
                                                "prediction_horizon_s": args.arch_horizon_s}})
    _arch_cfg = load_configs(args.config_name, labelspace=args.labelspace,
                             enable_lcd=False)
    arch_params = archetype_params_from_config(_arch_cfg)
    arch_horizon_s = archetype_horizon_from_config(_arch_cfg)

    pos, _, adj, trav = load_nav_graph(args.graph)
    _, nav_r, nav_nbr = load_nav_layer(args.graph)
    nav_edges = np.array([[i, j] for i, nb in enumerate(nav_nbr)
                          for j in nb if j > i], dtype=np.int32)
    xb = (pos[:, 0].min() - 6, pos[:, 0].max() + 6)
    yb = (pos[:, 1].min() - 6, pos[:, 1].max() + 6)
    opener = make_stream_opener(args, xb, yb)

    if args.mode == "predict":
        ok, err = pipeline.load_flow_state(args.flow_state)
        print(f"[run_viz] load_flow_state ok={ok} err={err!r}", flush=True)
        rec = run_predict(args, pipeline, arch_params, arch_horizon_s, pos, nav_r,
                          nav_edges, opener)
    else:
        if args.flow_state:
            ok, err = pipeline.load_flow_state(args.flow_state)
            print(f"[run_viz] warm-start load_flow_state ok={ok} err={err!r}", flush=True)
        poly, _ = build_coverage_loop(pos, adj, trav)
        sim = RobotSim(poly, pos, radius=args.fov_m)
        print(f"[run_viz] {len(pos)} cells, coverage loop {sim.total:.0f} m, "
              f"{len(args.days)} days", flush=True)
        rec = run_online(args, pipeline, arch_params, arch_horizon_s, pos, nav_r,
                         nav_edges, sim, opener)

    # Occupancy-map walls: static geometry authored from the floor plan. Each wall
    # cell inherits the coverage time of its nearest nav cell, so walls build up
    # alongside the free-cell floor as the exploration reaches them.
    if args.occupancy:
        occ = np.load(args.occupancy)
        wall_xy = occ["wall_xy"].astype(float)
        if len(wall_xy):
            nn = cKDTree(pos).query(wall_xy)[1]
            rec["wall_xy"] = wall_xy
            rec["wall_appear_t"] = np.asarray(rec["nav_appear_t"], float)[nn]
            print(f"[run_viz] walls: {len(wall_xy)} cells from {args.occupancy}",
                  flush=True)

    np.savez_compressed(args.out_npz, **rec)
    print(f"[run_viz] wrote {args.out_npz}: {len(rec['flow_times'])} frames, "
          f"{len(pos)} nodes, {len(rec['det_step'])} det arrows", flush=True)


if __name__ == "__main__":
    main()
