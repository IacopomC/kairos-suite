#!/usr/bin/env python3
r"""Read a frozen flow state out as a per-cell mixture, one file per hour.

Loads a trained ``flow_state.bin``, asks it for the mapped cells of the named
regions at one instant per hour of a chosen day, and writes each answer as a
map of dynamics: one row per mixture component, carrying the component's
direction, the cell's predicted speed, and the component's mixing weight at
that instant. ``plot_mod_panels.py`` draws those files; nothing else here
draws anything.

The state is queried through ``score_observations``, the same entry point the
evaluation scores detections with, so a drawn panel and a reported number come
from one forecast. The cells are the mapped voxels of the authored graph, and
the query needs no robot and no detections: the state is frozen and the time is
named, so any day (held out or not) can be read out.

Arrows at the voxel pitch are too dense to read over a room, so components are
summed into square bins of ``--pitch`` metres: a bin's weight for a direction
is the sum of its cells' weights for that direction, and its speed is their
mean. ``--pitch 0.4`` keeps the native cells.

Hours are hours of the local clock, ``--utc-offset-h`` east of UTC, and hour h
is answered at h:``--minute``.

Runs in the pipeline container (Python 3.8), where the bindings live.

Usage:
    python3 export_flow_mod.py --flow-state /out/train/flow_state.bin \
        --graph /data/atc/graph/atc_graph.json --scene-dir stub_scene \
        --config-name atc --date 2013-09-01 --utc-offset-h 9 \
        --hours 9-20 --region open=-20,0,-10,10 --region corridor=15,25,-17,-7 \
        --out-dir /out/qualitative
"""
import argparse
import calendar
import os
import pathlib
import sys

import numpy as np

from scene_loader import get_dataloader
from kairos_python.pipeline import load_pipeline

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "grounded_sim"))
from robot_sim import install_nav_layer, load_nav_graph  # noqa: E402

VOXEL_SIZE = 0.4


def parse_regions(values):
    """``name=x0,x1,y0,y1`` pairs as (name, box), in the order given."""
    out = []
    for v in values:
        name, box = v.split("=", 1)
        x0, x1, y0, y1 = (float(z) for z in box.split(","))
        out.append((name, (x0, x1, y0, y1)))
    return out


def parse_hours(spec):
    """``9-20`` or ``9,12,17``."""
    if "-" in spec:
        lo, hi = (int(v) for v in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(v) for v in spec.split(",")]


def bin_cells(x, y, weights, speeds, pitch):
    """Components summed into square bins of `pitch` metres.

    Returns (centre x, centre y, per-slot weight, mean speed) per non-empty
    bin. A bin's weight for a slot is the sum over its cells, so a direction
    two cells agree on carries twice one cell's weight, and the renderer's
    per-cell normalization leaves the shape of the mixture unchanged.
    """
    bx = np.floor(x / pitch).astype(np.int64)
    by = np.floor(y / pitch).astype(np.int64)
    keys, inv = np.unique(np.stack([bx, by], 1), axis=0, return_inverse=True)
    K = weights.shape[1]
    w = np.zeros((len(keys), K))
    rho = np.zeros(len(keys))
    n = np.zeros(len(keys))
    for i, b in enumerate(inv):
        w[b] += weights[i]
        rho[b] += speeds[i]
        n[b] += 1
    cx = (keys[:, 0] + 0.5) * pitch
    cy = (keys[:, 1] + 0.5) * pitch
    return cx, cy, w, rho / np.maximum(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-state", required=True)
    ap.add_argument("--graph", required=True, help="authored graph of mapped cells")
    ap.add_argument("--scene-dir", required=True, help="stub scene the sensor needs")
    ap.add_argument("--config-name", default="atc")
    ap.add_argument("--labelspace", default="ade20k_outdoor")
    ap.add_argument("--region", action="append", required=True,
                    help="name=x0,x1,y0,y1, repeatable")
    ap.add_argument("--date", required=True, help="the day to read out, ISO")
    ap.add_argument("--hours", default="9-20", help="local clock hours")
    ap.add_argument("--minute", type=int, default=30,
                    help="minute past the hour each panel answers for")
    ap.add_argument("--utc-offset-h", type=float, default=9.0,
                    help="hours east of UTC the local clock runs")
    ap.add_argument("--pitch", type=float, default=1.0,
                    help="bin size in metres; 0.4 keeps the native cells")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="metres kept outside each region, for edge arrows")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    regions = parse_regions(a.region)
    hours = parse_hours(a.hours)

    pos, _, _, _ = load_nav_graph(a.graph)
    keep = {}
    for name, (x0, x1, y0, y1) in regions:
        m = a.margin
        sel = np.where((pos[:, 0] >= x0 - m) & (pos[:, 0] <= x1 + m) &
                       (pos[:, 1] >= y0 - m) & (pos[:, 1] <= y1 + m))[0]
        keep[name] = sel
        print("[export] region {}: {} mapped cells".format(name, len(sel)), flush=True)
    order = np.concatenate([keep[n] for n, _ in regions])

    data = get_dataloader(a.scene_dir)
    pipeline = load_pipeline(data, a.config_name, a.labelspace, enable_lcd=False)
    if pipeline is None:
        raise SystemExit("[export] load_pipeline failed")
    ok, err = pipeline.load_flow_state(a.flow_state)
    if not ok:
        raise SystemExit("[export] load_flow_state failed: {}".format(err))
    install_nav_layer(pipeline, a.graph)
    print("[export] loaded {} cells".format(pipeline.flow_stats().get("cells")),
          flush=True)

    y, mo, d = (int(z) for z in a.date.split("-"))
    midnight = calendar.timegm((y, mo, d, 0, 0, 0, 0, 0, 0)) - a.utc_offset_h * 3600.0

    xyz = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in order]
    zeros = [0.0] * len(order)
    for hour in hours:
        t = midnight + hour * 3600.0 + a.minute * 60.0
        sc = pipeline.score_observations([t] * len(order), xyz, zeros, zeros, 0.0)
        matched = np.asarray(sc["matched"], bool)
        rho = np.asarray(sc["mean_rho"], float)
        slots = sc["slot_weights"]
        K = max((len(s) for s in slots), default=0)
        if K == 0:
            print("[export] hour {}: no cell answered".format(hour), flush=True)
            continue
        W = np.zeros((len(order), K))
        for i, s in enumerate(slots):
            if matched[i] and len(s) == K:
                W[i] = s
        # A cell whose weights are all zero carries no mixture to draw.
        live = matched & (W.sum(axis=1) > 0)
        slot_angle = np.arange(K) * (2.0 * np.pi / K)
        base = 0
        for name, _ in regions:
            n = len(keep[name])
            sl = slice(base, base + n)
            base += n
            idx = np.where(live[sl])[0]
            if not len(idx):
                print("[export] {} hour {}: no cell answered".format(name, hour),
                      flush=True)
                continue
            px = np.asarray([xyz[sl.start + i][0] for i in idx])
            py = np.asarray([xyz[sl.start + i][1] for i in idx])
            cx, cy, w, speed = bin_cells(px, py, W[sl][idx], rho[sl][idx],
                                         max(a.pitch, VOXEL_SIZE))
            out = os.path.join(a.out_dir, name)
            if not os.path.isdir(out):
                os.makedirs(out)
            path = os.path.join(out, "{:02d}.csv".format(hour))
            with open(path, "w") as f:
                f.write("x,y,mean_speed,mean_motion_angle,weight\n")
                for j in range(len(cx)):
                    for k in range(K):
                        f.write("{:.3f},{:.3f},{:.4f},{:.6f},{:.6f}\n".format(
                            cx[j], cy[j], speed[j], slot_angle[k], w[j, k]))
            print("[export] {} -> {} bins, {} cells answered".format(
                path, len(cx), len(idx)), flush=True)


if __name__ == "__main__":
    main()
