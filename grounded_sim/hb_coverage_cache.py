#!/usr/bin/env python3
r"""Build the coverage-counts cache that author_graph.py consumes, from the
HB station stream.

author_graph.py is dataset-agnostic except for its coverage pass (ATC npz
files). Its --coverage-cache npz short-circuits that pass when the cached
`days` list and voxel size match the arguments, so this script bins the
10 Hz interpolated training stream per 0.4 m cell over the map extent and
writes that cache; author_graph.py is then invoked with the same date list
and never opens a day file. The interpolated stream (the one training
consumes) defines coverage, so the authored region matches what the flow
model can observe.

    python3 hb_coverage_cache.py --hb-root .../Data-2 --station HB \
        --map-yaml .../hb_occupancy.yaml --dates <ISO dates> --out cache.npz
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hb_stream import HBDayStream, hb_day_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hb-root", required=True)
    ap.add_argument("--station", choices=["HB", "PB"], required=True)
    ap.add_argument("--map-yaml", required=True)
    ap.add_argument("--dates", nargs="+", required=True,
                    help="TRAINING dates (ISO); never pass test dates")
    ap.add_argument("--voxel-size", type=float, default=0.4)
    ap.add_argument("--interp-hz", type=float, default=10.0)
    ap.add_argument("--gap-max-s", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = origin = None
    dims = None
    with open(args.map_yaml) as f:
        for line in f:
            if line.startswith("resolution:"):
                res = float(line.split(":")[1])
            if line.startswith("origin:"):
                origin = [float(v) for v in
                          line.split("[")[1].split("]")[0].split(",")]
            if line.startswith("image:"):
                pgm = pathlib.Path(args.map_yaml).parent / line.split(":")[1].strip()
    with open(pgm, "rb") as f:
        assert f.readline().strip() == b"P5"
        w, h = map(int, f.readline().split())
    ext = (origin[0], origin[0] + w * res, origin[1], origin[1] + h * res)
    x0, x1, y0, y1 = ext
    vs = args.voxel_size
    nx = int(np.ceil((x1 - x0) / vs))
    ny = int(np.ceil((y1 - y0) / vs))
    counts = np.zeros((nx, ny), dtype=np.int64)
    print(f"[hb-coverage] {args.station}: extent x=({x0},{x1}) "
          f"y=({y0},{y1}) -> {nx}x{ny} cells, {len(args.dates)} dates")
    for i, d in enumerate(sorted(args.dates)):
        files = hb_day_files(args.hb_root, args.station, d)
        s = HBDayStream(files, x_bounds=(x0, x1), y_bounds=(y0, y1),
                           interp_hz=args.interp_hz, gap_max_s=args.gap_max_s)
        ix = ((s.x - x0) / vs).astype(np.int64)
        iy = ((s.y - y0) / vs).astype(np.int64)
        np.add.at(counts, (ix, iy), 1)
        if i % 20 == 0:
            print(f"  {d} ({i+1}/{len(args.dates)}): {s.t.size} samples",
                  flush=True)
    np.savez_compressed(args.out, counts=counts, voxel_size=vs,
                        days=[str(d) for d in sorted(args.dates)])
    print(f"[hb-coverage] cached -> {args.out} "
          f"(covered cells: {(counts > 0).sum()})")


if __name__ == "__main__":
    main()
