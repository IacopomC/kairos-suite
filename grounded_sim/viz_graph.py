#!/usr/bin/env python3
r"""Render a spark_dsg scene graph as a layered 3D plot plus a top-down 2D plot.

Dataset-agnostic: takes any graph.json. Layers are stacked by z-offset and nodes
coloured by type (places / rooms / objects), edges drawn within and between
layers --- the same layered-3DSG view used elsewhere. Runs in a container that
has spark_dsg + matplotlib (kairos-suite).

Usage:
    python3 viz_graph.py --graph /data/atc/graph/atc_graph.json \
        --out-prefix figs/atc_graph

Writes a layered 3D plot and a top-down 2D plot. Useful for checking an
authored graph before training on it.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import spark_dsg as dsg

# layer -> (z-offset, colour, marker size); colours follow the usual DSG palette
LAYERS = [
    (dsg.DsgLayers.PLACES, "places", 0.0, "#AB63FA", 6),
    (dsg.DsgLayers.OBJECTS, "objects", 6.0, "#00CC96", 20),
    (dsg.DsgLayers.ROOMS, "rooms", 12.0, "#EF553B", 60),
]


def collect(graph):
    """Return {layer_name: {"z":z,"color":c,"size":s,"pos":{id:(x,y)}}} and a
    global id -> (x,y,z) map for edge drawing."""
    info, gpos = {}, {}
    for layer_id, name, z, color, size in LAYERS:
        if not graph.has_layer(layer_id):
            continue
        layer = graph.get_layer(layer_id)
        pos = {}
        for n in layer.nodes:
            p = n.attributes.position
            # node.id is a NodeSymbol; .value is the uint64 edges reference.
            nid = n.id.value
            pos[nid] = (float(p[0]), float(p[1]))
            gpos[nid] = (float(p[0]), float(p[1]), z)
        if pos:
            info[name] = dict(z=z, color=color, size=size, pos=pos,
                              layer=layer, layer_id=layer_id)
    return info, gpos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    graph = dsg.DynamicSceneGraph.load(args.graph)
    print(f"[viz] {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    info, gpos = collect(graph)
    for name, d in info.items():
        print(f"[viz]   {name}: {len(d['pos'])} nodes")

    # ---- 3D layered view ----
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection="3d")
    for name, d in info.items():
        xs = [p[0] for p in d["pos"].values()]
        ys = [p[1] for p in d["pos"].values()]
        zs = [d["z"]] * len(xs)
        ax.scatter(xs, ys, zs, s=d["size"], c=d["color"], label=name,
                   depthshade=False, edgecolors="none")
        segs = []
        for e in d["layer"].edges:
            if e.source in gpos and e.target in gpos:
                a, b = gpos[e.source], gpos[e.target]
                segs.append([(a[0], a[1], d["z"]), (b[0], b[1], d["z"])])
        if segs:
            ax.add_collection3d(Line3DCollection(segs, colors="#cccccc",
                                                 linewidths=0.3, alpha=0.5))
    # interlayer edges (e.g. room -> place)
    ilsegs = [[gpos[e.source], gpos[e.target]] for e in graph.interlayer_edges
              if e.source in gpos and e.target in gpos]
    if ilsegs:
        ax.add_collection3d(Line3DCollection(ilsegs, colors="#EF553B",
                                             linewidths=0.2, alpha=0.25))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("layer")
    ax.legend(loc="upper right")
    ax.view_init(elev=55, azim=-60)
    fig.savefig(f"{args.out_prefix}_3d.png", dpi=100, bbox_inches="tight")
    print(f"[viz] wrote {args.out_prefix}_3d.png")

    # ---- 2D top-down (places coloured by parent room) ----
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    if "places" in info:
        d = info["places"]
        # place-place edges
        segs = [[d["pos"][e.source], d["pos"][e.target]]
                for e in d["layer"].edges
                if e.source in d["pos"] and e.target in d["pos"]]
        ax2.add_collection(LineCollection(segs, colors="#dddddd", linewidths=0.4))
        # colour each place by its parent room id (fallback: single colour)
        ids = list(d["pos"].keys())
        parent = {}
        for n in d["layer"].nodes:
            p = n.get_parent()
            pv = getattr(p, "value", p)
            parent[n.id.value] = pv if pv is not None else -1
        uniq = sorted(set(parent.values()))
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
        cidx = {r: cmap[i % len(cmap)] for i, r in enumerate(uniq)}
        xs = [d["pos"][i][0] for i in ids]
        ys = [d["pos"][i][1] for i in ids]
        cs = [cidx[parent[i]] for i in ids]
        ax2.scatter(xs, ys, s=8, c=cs)
    if "rooms" in info:
        d = info["rooms"]
        rx = [p[0] for p in d["pos"].values()]
        ry = [p[1] for p in d["pos"].values()]
        ax2.scatter(rx, ry, s=140, marker="*", c="#EF553B",
                    edgecolors="black", linewidths=0.5, zorder=5, label="rooms")
    ax2.set_aspect("equal"); ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    ax2.set_title("ATC 3DSG (top-down): places coloured by room, edges = adjacency")
    ax2.legend(loc="upper right")
    fig2.savefig(f"{args.out_prefix}_top.png", dpi=110, bbox_inches="tight")
    print(f"[viz] wrote {args.out_prefix}_top.png")


if __name__ == "__main__":
    main()
