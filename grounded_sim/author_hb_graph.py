#!/usr/bin/env python3
r"""Author the HB-station TSDF-voxel graph from training coverage.

The graph is the instrumented region alone, without wall geometry: one PLACES node per 0.4 m cell that holds at least one training
detection of the interpolated stream (the same stream the flow model trains
on), 8-adjacent edges, every node traversable. The graph exists to give the
trainer and the evaluator their mapped-cell universe (set_visible_voxels /
presence scan) in full-visibility runs. The patrol routes on the graph from
`author_graph.py` instead.

Coverage uses the TRAINING dates only, and only dates covered by every
camera of the station. A sidecar npz records the cell set and parameters.

Runs in a container with spark_dsg (kairos-suite).

Usage:
    python3 author_hb_graph.py --station HB \
        --train-start 2022-09-05 --train-end 2023-04-23 \
        --out $DATA_ROOT/hb-station/graph/hb_graph.json

Training dates only, for the same reason as author_graph.py. Only dates covered
by every camera of the station are used, so spatial coverage is constant across
the span.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import spark_dsg as dsg
from spark_dsg._dsg_bindings import NodeSymbol

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hb_stream import HBDayStream, station_dates

STATION_BOUNDS = {"HB": ([-2.0, 35.0], [-32.0, 2.0]),
                  "PB": ([-4.0, 82.0], [-22.0, 4.0])}
VS = 0.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/hb-station/Data-2")
    ap.add_argument("--station", choices=["HB", "PB"], required=True)
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--interp-hz", type=float, default=10.0)
    ap.add_argument("--gap-max-s", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xb, yb = STATION_BOUNDS[args.station]
    dates = {d: f for d, f in station_dates(args.root, args.station).items()
             if args.train_start <= d <= args.train_end}
    print(f"[author-hb] {args.station}: {len(dates)} all-camera train dates "
          f"{min(dates)}..{max(dates)}")

    OFF = 1 << 20            # world cell indices may be negative near the origin
    cells = set()
    for d, files in dates.items():
        s = HBDayStream(files, x_bounds=xb, y_bounds=yb,
                           interp_hz=args.interp_hz, gap_max_s=args.gap_max_s)
        vx = np.floor(s.x / VS).astype(np.int64)
        vy = np.floor(s.y / VS).astype(np.int64)
        for k in np.unique((vx + OFF) * (1 << 21) + (vy + OFF)):
            cells.add(int(k))
        print(f"  [coverage] {d}: {s.t.size} detections, {len(cells)} cells")
    keys = sorted(cells)
    kvx = np.array([(k >> 21) - OFF for k in keys], dtype=np.int64)
    kvy = np.array([(k & ((1 << 21) - 1)) - OFF for k in keys], dtype=np.int64)

    G = dsg.DynamicSceneGraph()
    idx_of = {}
    for i, (ix, iy) in enumerate(zip(kvx.tolist(), kvy.tolist())):
        a = dsg.PlaceNodeAttributes()
        a.position = np.array([(ix + 0.5) * VS, (iy + 0.5) * VS, 0.0],
                              dtype=np.float64)
        a.distance = VS          # > 0: uniformly traversable, no wall grid
        a.num_basis_points = 0
        G.add_node(dsg.DsgLayers.PLACES, NodeSymbol("p", i).value, a)
        idx_of[(ix, iy)] = i
    n_edges = 0
    for (ix, iy), i in idx_of.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            j = idx_of.get((ix + dx, iy + dy))
            if j is not None:
                G.insert_edge(NodeSymbol("p", i).value, NodeSymbol("p", j).value)
                n_edges += 1
    G.save(args.out)
    np.savez_compressed(args.out + ".cells.npz", vx=kvx, vy=kvy,
                        voxel_size=VS, x_bounds=xb, y_bounds=yb,
                        interp_hz=args.interp_hz, gap_max_s=args.gap_max_s,
                        train_start=args.train_start, train_end=args.train_end,
                        dates=sorted(dates))
    print(f"[author-hb] saved {args.out}: {len(keys)} nodes, {n_edges} edges "
          f"(+ .cells.npz sidecar)")


if __name__ == "__main__":
    main()
