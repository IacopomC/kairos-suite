r"""Draw the embodied robot's coverage loop (the exact polyline the trainer
builds via build_coverage_loop on the authored graph) over the dataset
occupancy map, with a detection subsample underneath as an orientation gate.

    python3 draw_patrol.py --graph graph.json --map-pgm map.pgm \
        --map-yaml map.yaml --overlay-npz day.npz --out patrol.png
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from robot_sim import build_coverage_loop


def load_nav_graph_json(graph_path):
    """load_nav_graph without the scene-graph bindings: parse the authored
    graph json directly (same fields, same outputs)."""
    import json
    from scipy.sparse import csr_matrix
    doc = json.load(open(graph_path))
    nodes = [n for n in doc["nodes"] if int(n["layer"]) == 3]
    node_ids = [int(n["id"]) for n in nodes]
    idx_of = {v: i for i, v in enumerate(node_ids)}
    pos = np.array([n["attributes"]["position"][:2] for n in nodes])
    trav = np.array([float(n["attributes"].get("distance", 0.0)) > 0
                     for n in nodes], bool)
    rows, cols, w = [], [], []
    for e in doc["edges"]:
        s, t = int(e["source"]), int(e["target"])
        if s in idx_of and t in idx_of:
            i, j = idx_of[s], idx_of[t]
            d = float(np.hypot(*(pos[i] - pos[j])))
            rows += [i, j]; cols += [j, i]; w += [d, d]
    adj = csr_matrix((w, (rows, cols)), shape=(len(node_ids), len(node_ids)))
    return pos, node_ids, adj, trav


def load_map(pgm, yaml_path):
    img = np.asarray(Image.open(pgm))
    meta = {}
    for line in pathlib.Path(yaml_path).read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    res = float(meta["resolution"])
    origin = [float(x) for x in meta["origin"].strip("[]").split(",")[:2]]
    h, w = img.shape[:2]
    extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]
    return img, extent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--map-pgm", required=True)
    ap.add_argument("--map-yaml", required=True)
    ap.add_argument("--overlay-npz", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pos, node_ids, adj, trav = load_nav_graph_json(args.graph)
    poly, order = build_coverage_loop(pos, adj, trav)
    poly = np.asarray(poly)
    seg = np.linalg.norm(np.diff(poly[:, :2], axis=0), axis=1)
    length = float(seg.sum())

    img, extent = load_map(args.map_pgm, args.map_yaml)
    ar = (extent[1] - extent[0]) / (extent[3] - extent[2])
    fig, ax = plt.subplots(figsize=(10, 10 / max(ar, 0.4)))
    # ROS map convention: the pgm's first row is the TOP of the map.
    ax.imshow(img, cmap="gray", extent=extent, origin="upper",
              vmin=0, vmax=255, interpolation="nearest")
    if args.overlay_npz:
        z = np.load(args.overlay_npz)
        step = max(1, len(z["t"]) // 60000)
        ax.scatter(z["x"][::step] / 1000.0, z["y"][::step] / 1000.0,
                   s=0.3, c="#77bbff", alpha=0.25, zorder=2, linewidths=0)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    pts = ax.scatter(poly[:, 0], poly[:, 1], c=s, cmap="viridis", s=1.2,
                     zorder=4, linewidths=0)
    ax.plot(poly[:, 0], poly[:, 1], color="black", lw=0.3, alpha=0.5, zorder=3)
    ax.scatter(*poly[0, :2], marker="o", s=120, c="red", zorder=6)
    ax.annotate("start", poly[0, :2], xytext=(6, 6),
                textcoords="offset points", fontsize=11, color="red",
                fontweight="bold")
    fig.colorbar(pts, ax=ax, shrink=0.7, label="arclength along the loop [m]")
    ax.set_title(f"embodied coverage loop: {length:.0f} m, "
                 f"{len(poly)} waypoints, revisit {length/1.5/60:.1f} min "
                 f"at 1.5 m/s")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"loop {length:.1f} m, {len(poly)} pts -> {args.out}")


if __name__ == "__main__":
    main()
