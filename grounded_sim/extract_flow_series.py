#!/usr/bin/env python3
r"""Extract a time-series of the learned flow for Rerun logging.

Queries the trained flow_state at a sequence of times across the session,
aggregates per-voxel flow onto the medial-axis nav nodes (nearest node within
its clearance radius, computed once since it is static), classifies each nav
node's motion archetype, and records the robot pose along its patrol. Dumps a
portable .npz that rerun_log.py (any Python + rerun) turns into a recording.

Runs in the pipeline container (py3.8, flow bindings).

Usage:
    python3 extract_flow_series.py --graph <atc_graph.json> \
        --nav-graph <nav_graph.json> --csv <robot_path.csv> \
        --scene-dir stub_scene --flow-state OUT/train/flow_state.bin \
        --out series.npz
"""
import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scene_loader import get_dataloader
from kairos_python import flow_archetype
from kairos_python.pipeline import (
    load_pipeline, load_configs, archetype_params_from_config)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from robot_sim import (install_nav_layer, load_nav_graph, build_patrol, RobotSim,
                       ROBOT_SPEED_MPS, ACTIVE_WINDOW_RADIUS_M)
from flow_viz import classify, load_nav_layer, K, SLOT_ANG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--nav-graph", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--config-name", default="tbd")
    ap.add_argument("--labelspace", default="ade20k_outdoor")
    ap.add_argument("--flow-state", required=True)
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--n-flow-steps", type=int, default=45,
                    help="flow query snapshots across the session")
    ap.add_argument("--robot-dt", type=float, default=1.0,
                    help="robot-pose sampling period (s); small = smooth motion")
    args = ap.parse_args()

    vox, _, vadj, _ = load_nav_graph(args.graph)
    nav_pos, nav_r, nav_nbr = load_nav_layer(args.nav_graph)
    nav_edges = np.array([[i, j] for i, nbrs in enumerate(nav_nbr)
                          for j in nbrs if j > i], dtype=np.int32)
    poly, _ = build_patrol(vox, vadj, root=0)
    sim = RobotSim(poly, vox)
    print(f"[extract] {len(vox)} voxels, {len(nav_pos)} nav nodes, "
          f"patrol {sim.total:.0f} m")

    # static voxel -> nav assignment (nearest nav within its clearance radius)
    navtree = cKDTree(nav_pos)
    d, nn = navtree.query(vox)
    nav_members = [[] for _ in range(len(nav_pos))]
    for vi in range(len(vox)):
        if d[vi] <= nav_r[nn[vi]]:
            nav_members[nn[vi]].append(vi)

    t = pd.read_csv(args.csv, header=None, usecols=[0], dtype=np.float64).iloc[:, 0].values
    t0, t1 = float(t.min()), float(t.max())
    flow_times = np.linspace(t0, t1, args.n_flow_steps)

    data = get_dataloader(args.scene_dir)
    pipeline = load_pipeline(data, args.config_name, args.labelspace, enable_lcd=False)
    arch_params = archetype_params_from_config(
        load_configs(args.config_name, labelspace=args.labelspace, enable_lcd=False))
    ok, err = pipeline.load_flow_state(args.flow_state)
    print(f"[extract] load_flow_state ok={ok} err={err!r}")
    install_nav_layer(pipeline, args.graph)

    N, M = len(vox), len(nav_pos)
    xyz = [[float(vox[i, 0]), float(vox[i, 1]), 0.0] for i in range(N)]
    nav_theta = np.zeros((len(flow_times), M))
    nav_rho = np.zeros((len(flow_times), M))
    nav_active = np.zeros((len(flow_times), M), bool)
    nav_arch = np.zeros((len(flow_times), M), np.int8)
    nav_slots = np.zeros((len(flow_times), M, K))   # aggregated slot weights

    for ti, tq in enumerate(flow_times):
        sc = pipeline.score_observations([float(tq)] * N, xyz, [0.0] * N, [0.0] * N, 0.0)
        matched = np.array(sc["matched"], bool)
        mth = np.array(sc["mean_theta"], float)
        mrho = np.array(sc["mean_rho"], float)
        slots = sc["slot_weights"]
        Wv = np.zeros((N, K))
        for i in np.where(matched)[0]:
            if len(slots[i]) == K:
                Wv[i] = slots[i]
        for j, mem in enumerate(nav_members):
            mem_m = [i for i in mem if matched[i]]
            if not mem_m:
                continue
            w = Wv[mem_m].sum(axis=0)
            rho = float(mrho[mem_m].mean())
            vx = float(np.sum(np.cos(mth[mem_m]) * mrho[mem_m]))
            vy = float(np.sum(np.sin(mth[mem_m]) * mrho[mem_m]))
            nav_theta[ti, j] = np.arctan2(vy, vx)
            nav_rho[ti, j] = rho
            nav_active[ti, j] = True
            nav_arch[ti, j] = flow_archetype.CODE[classify(w, len(mem_m), arch_params)]
            nav_slots[ti, j] = w
        if ti % 10 == 0:
            print(f"  flow step {ti}/{len(flow_times)} active={int(nav_active[ti].sum())}")

    # robot pose sampled finely along the patrol (continuous motion)
    robot_times = np.arange(t0, t1, args.robot_dt)
    robot_xy = np.array([sim.pose_at_arclength(ROBOT_SPEED_MPS * (rt - t0))
                         for rt in robot_times])

    np.savez_compressed(
        args.out,
        t0=t0, t1=t1,
        vox=vox, nav_pos=nav_pos, nav_r=nav_r, nav_edges=nav_edges,
        room=np.array([vox[:, 0].mean(), vox[:, 1].mean()]),
        patrol=poly,
        flow_times=flow_times, nav_theta=nav_theta, nav_rho=nav_rho,
        nav_active=nav_active, nav_arch=nav_arch, nav_slots=nav_slots,
        robot_times=robot_times, robot_xy=robot_xy,
        arch_names=np.array(flow_archetype.NAMES),
        arch_colors=np.array(flow_archetype.color_list()),
        active_window_radius=ACTIVE_WINDOW_RADIUS_M,
    )
    print(f"[extract] wrote {args.out}: {len(flow_times)} flow steps, "
          f"{len(robot_times)} robot poses")


if __name__ == "__main__":
    main()
