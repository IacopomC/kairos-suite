#!/usr/bin/env python3
r"""Layered visualization of the learned flow on the ATC 3DSG.

    z3  room ................. one concourse
    z2  archetypes ........... adjacent same-class nav nodes grouped (motion class)
    z1  navigational nodes ... medial-axis (GVD) nodes with clearance radius
    z0  voxels (0.4 m) ....... SW-GMM flow  -> ARROWS

Voxels carry the flow (queried per 0.4 m cell -> arrows). Navigational nodes are
the medial-axis GVD nodes; each voxel is aggregated to its nearest nav node
within that node's clearance radius (many-to-one, matching the live aggregation).
Nav nodes are classified into motion archetypes and adjacent same-class nodes
grouped. Snapshot at --time-frac (flow is time-varying).

Usage:
    python3 flow_viz.py --graph <atc_graph.json> --nav-graph <nav_graph.json> \
        --csv <robot_path.csv> --scene-dir stub_scene \
        --flow-state OUT/train/flow_state.bin --out-prefix figs/flow

Also imported as a library by run_viz.py and extract_flow_series.py for its
layer rendering and archetype classification.
"""
import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree
import spark_dsg as dsg

from scene_loader import get_dataloader
from kairos_python import flow_archetype
from kairos_python._kairos_bindings import classify_archetype
from kairos_python.pipeline import (
    load_pipeline, load_configs, archetype_params_from_config)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from robot_sim import install_nav_layer, load_nav_graph

K = 8
SLOT_ANG = np.arange(K) * (2 * np.pi / K)
# name -> hex, mirroring the C++ ArchetypeType colours (single source).
ARCH_COLOR = dict(zip(flow_archetype.NAMES, flow_archetype.color_list()))


def classify(weights, support, params):
    """Archetype NAME via the shared C++ classifier. Weights are the aggregated
    slot weights (normalised here to the mixture weights the threshold expects);
    `support` is the voxel-member count; probability is bypassed (1.0) since this
    offline aggregation carries no per-node occupancy probability."""
    w = np.asarray(weights, float)
    s = w.sum()
    wn = (w / s).tolist() if s > 0 else w.tolist()
    return flow_archetype.NAMES[
        classify_archetype(1.0, int(support), wn, **params)]


def load_nav_layer(path):
    """nav nodes: positions (Nx2), clearance radius (N), neighbour lists."""
    G = dsg.DynamicSceneGraph.load(path)
    L = G.get_layer(dsg.DsgLayers.PLACES)
    nodes = list(L.nodes)
    ids = [n.id.value for n in nodes]
    idx = {v: i for i, v in enumerate(ids)}
    pos = np.array([[n.attributes.position[0], n.attributes.position[1]] for n in nodes])
    rad = np.array([float(n.attributes.distance) for n in nodes])
    nbr = [[] for _ in ids]
    for e in L.edges:
        if e.source in idx and e.target in idx:
            nbr[idx[e.source]].append(idx[e.target])
            nbr[idx[e.target]].append(idx[e.source])
    return pos, rad, nbr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="0.4 m voxel graph")
    ap.add_argument("--nav-graph", required=True, help="medial-axis nav graph")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--config-name", default="tbd")
    ap.add_argument("--labelspace", default="ade20k_outdoor")
    ap.add_argument("--flow-state", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--time-frac", type=float, default=0.5)
    args = ap.parse_args()

    vox, _, _, _ = load_nav_graph(args.graph)            # 0.4 m voxel centres
    nav_pos, nav_r, nav_nbr = load_nav_layer(args.nav_graph)
    print(f"[flow_viz] {len(vox)} voxels, {len(nav_pos)} nav nodes")

    t = pd.read_csv(args.csv, header=None, usecols=[0], dtype=np.float64).iloc[:, 0].values
    t_query = float(t.min() + args.time_frac * (t.max() - t.min()))
    print(f"[flow_viz] query {pd.to_datetime(t_query, unit='s')} UTC")

    data = get_dataloader(args.scene_dir)
    pipeline = load_pipeline(data, args.config_name, args.labelspace, enable_lcd=False)
    arch_params = archetype_params_from_config(
        load_configs(args.config_name, labelspace=args.labelspace, enable_lcd=False))
    ok, err = pipeline.load_flow_state(args.flow_state)
    print(f"[flow_viz] load_flow_state ok={ok} err={err!r}")
    install_nav_layer(pipeline, args.graph)

    N = len(vox)
    xyz = [[float(vox[i, 0]), float(vox[i, 1]), 0.0] for i in range(N)]
    sc = pipeline.score_observations([t_query] * N, xyz, [0.0] * N, [0.0] * N, 0.0)
    matched = np.array(sc["matched"], dtype=bool)
    mth = np.array(sc["mean_theta"], float)
    mrho = np.array(sc["mean_rho"], float)
    slots = sc["slot_weights"]
    W = np.zeros((N, K))
    for i in np.where(matched)[0]:
        if len(slots[i]) == K:
            W[i] = slots[i]
    print(f"[flow_viz] matched voxels: {int(matched.sum())}/{N}")

    # --- aggregate voxels -> nearest nav node within its clearance radius ---
    navtree = cKDTree(nav_pos)
    mi = np.where(matched)[0]
    d, nn = navtree.query(vox[mi])
    assign = -np.ones(N, dtype=int)                    # voxel -> nav index
    for vi, j, dist in zip(mi, nn, d):
        if dist <= nav_r[j]:
            assign[vi] = j
    nav_members = [[] for _ in range(len(nav_pos))]
    for vi in mi:
        if assign[vi] >= 0:
            nav_members[assign[vi]].append(vi)

    nav_arch = np.array([""] * len(nav_pos), dtype=object)
    nav_th = np.zeros(len(nav_pos)); nav_rho = np.zeros(len(nav_pos))
    for j, mem in enumerate(nav_members):
        if not mem:
            continue
        w = W[mem].sum(axis=0)
        rho = float(mrho[mem].mean())
        vx = float(np.sum(np.cos(mth[mem]) * mrho[mem]))
        vy = float(np.sum(np.sin(mth[mem]) * mrho[mem]))
        nav_th[j] = np.arctan2(vy, vx); nav_rho[j] = rho
        nav_arch[j] = classify(w, len(mem), arch_params)
    active = np.array([len(m) > 0 for m in nav_members])
    counts = {c: int((nav_arch == c).sum()) for c in ARCH_COLOR}
    supp = [len(m) for m in nav_members if m]
    print(f"[flow_viz] active nav nodes: {int(active.sum())}/{len(nav_pos)}; "
          f"voxel support per node: median={int(np.median(supp)) if supp else 0}, "
          f"max={max(supp) if supp else 0}  archetypes={counts}")

    # voxel colour = archetype of its assigned nav node
    vox_arch = np.array([""] * N, dtype=object)
    for vi in mi:
        if assign[vi] >= 0:
            vox_arch[vi] = nav_arch[assign[vi]]

    # archetypes: connected components of same-class adjacent nav nodes
    seen = np.zeros(len(nav_pos), bool)
    arche = []
    for i in np.where(active)[0]:
        if seen[i]:
            continue
        cls = nav_arch[i]; stack, comp = [i], []; seen[i] = True
        while stack:
            c = stack.pop(); comp.append(c)
            for j in nav_nbr[c]:
                if active[j] and not seen[j] and nav_arch[j] == cls:
                    seen[j] = True; stack.append(j)
        arche.append(dict(x=float(nav_pos[comp, 0].mean()),
                          y=float(nav_pos[comp, 1].mean()),
                          arch=cls, members=comp, size=len(comp)))
    room = (float(vox[mi, 0].mean()), float(vox[mi, 1].mean()))

    def node_arrows(ax_, z=None):
        # arrows are the aggregated node dynamics (one per navigational node),
        # not per-voxel; length ~ node mean speed, colour = node archetype.
        for cls, color in ARCH_COLOR.items():
            m = active & (nav_arch == cls)
            if not m.any():
                continue
            L_ = np.clip(nav_rho[m], 0.1, 2.0) * 1.6
            u, v = np.cos(nav_th[m]) * L_, np.sin(nav_th[m]) * L_
            if z is None:
                ax_.quiver(nav_pos[m, 0], nav_pos[m, 1], u, v, color=color,
                           angles="xy", scale_units="xy", scale=1.0, width=0.0022,
                           headwidth=4, headlength=5, alpha=0.95, zorder=4)
            else:
                ax_.quiver(nav_pos[m, 0], nav_pos[m, 1], np.full(m.sum(), z), u, v,
                           np.zeros(m.sum()), color=color, alpha=0.9, linewidth=1.0)

    # ---------- top-down ----------
    fig, ax = plt.subplots(figsize=(18, 9))
    # voxels are faint support underneath, tinted by their nav node's archetype
    ax.scatter(vox[:, 0], vox[:, 1], s=2, c="#eeeeee", zorder=0)
    for cls, color in ARCH_COLOR.items():
        m = matched & (vox_arch == cls)
        if m.any():
            ax.scatter(vox[m, 0], vox[m, 1], s=2, c=color, alpha=0.18, zorder=1)
    node_arrows(ax)                                    # arrows = node dynamics
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"learned flow @ {pd.to_datetime(t_query, unit='s')} UTC "
                 f"(arrows=navigational-node dynamics, colour=archetype; "
                 f"voxels=faint support)")
    ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=f"{k} ({counts[k]})")
                       for k, c in ARCH_COLOR.items()], loc="upper right")
    fig.savefig(f"{args.out_prefix}_flow_top.png", dpi=120, bbox_inches="tight")
    print(f"[flow_viz] wrote {args.out_prefix}_flow_top.png")

    # ---------- layered 3D ----------
    Z = {"voxels": 0.0, "nav": 7.0, "archetypes": 14.0, "room": 21.0}
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection="3d")
    # z0 voxels: support points tinted by their nav node's archetype (no arrows)
    for cls, color in ARCH_COLOR.items():
        m = matched & (vox_arch == cls)
        if m.any():
            ax.scatter(vox[m, 0], vox[m, 1], np.full(m.sum(), Z["voxels"]),
                       s=2, c=color, alpha=0.35, depthshade=False, edgecolors="none")
    # z1 navigational nodes (sized by clearance radius) + aggregated arrows
    ai = np.where(active)[0]
    ax.scatter(nav_pos[ai, 0], nav_pos[ai, 1], np.full(len(ai), Z["nav"]),
               s=6 + nav_r[ai] * 6, c=[ARCH_COLOR[nav_arch[j]] for j in ai],
               depthshade=False, edgecolors="none")
    node_arrows(ax, z=Z["nav"])
    aedges = []
    for a in arche:
        c = ARCH_COLOR[a["arch"]]
        ax.scatter([a["x"]], [a["y"]], [Z["archetypes"]], s=40 + 5 * a["size"],
                   c=c, edgecolors="black", linewidths=0.4, depthshade=False)
        for m in a["members"]:
            aedges.append([(nav_pos[m, 0], nav_pos[m, 1], Z["nav"]),
                           (a["x"], a["y"], Z["archetypes"])])
        aedges.append([(a["x"], a["y"], Z["archetypes"]), (room[0], room[1], Z["room"])])
    ax.add_collection3d(Line3DCollection(aedges, colors="#cccccc",
                                         linewidths=0.2, alpha=0.3))
    ax.scatter([room[0]], [room[1]], [Z["room"]], s=220, marker="*",
               c="#EF553B", edgecolors="black", depthshade=False)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_zticks(list(Z.values())); ax.set_zticklabels(list(Z.keys()))
    ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=k)
                       for k, c in ARCH_COLOR.items()], loc="upper right")
    ax.view_init(elev=30, azim=-60)
    fig.savefig(f"{args.out_prefix}_flow_3d.png", dpi=110, bbox_inches="tight")
    print(f"[flow_viz] wrote {args.out_prefix}_flow_3d.png")


if __name__ == "__main__":
    main()
