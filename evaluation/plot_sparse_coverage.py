#!/usr/bin/env python3
r"""Per-cell CRPS spatial heatmaps for the sparse-embodied-coverage study.

Reads the ``cellmap.npz`` dumps written by ``evaluate.py`` (per-cell circular
CRPS keyed to voxel indices) for each method's budgeted-patrol run, averages
over seeds, and writes one standalone image per panel, painted over the ATC
occupancy map. The absolute maps share one sequential blue scale so they tile
comparably; each difference map uses its own diverging scale (blue = Kairos
better).

Outputs (``.png`` and ``.pdf`` each) into ``--out-dir``: one
``sparse_coverage_<method>`` per method, plus one
``sparse_coverage_diff_kairos_<method>`` per comparison method.

Example:
    python3 plot_sparse_coverage.py \
        --results-dir ../results/exploration \
        --map-pgm /data/atc/ATC-map/localization_grid.pgm \
        --map-yaml /data/atc/ATC-map/localization_grid.yaml \
        --out-dir .
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

# method -> cellmap directory prefix (dir is "<prefix>_<policy>_s<seed>")
PREFIX = {"kairos": "k"}
LABEL = {"kairos": "Kairos"}
# Comparison methods are added at the command line (see --compare); their runs
# are produced outside this suite.

# Shared sequential scale for the absolute maps. Blue for low CRPS, white for
# high, matching the difference maps where blue marks the better forecast. Each
# difference map takes its own diverging limit, since the spread depends on the
# method compared against.
SEQ_CMAP = "Blues_r"
SEQ_LO, SEQ_HI = 0.40, 1.50
DIFF_MAX = {}          # per-method colour limit for the difference map
DIFF_MAX_DEFAULT = 1.00


def read_yaml_map(pgm_path, yaml_path):
    res, origin = 0.05, (0.0, 0.0)
    with open(yaml_path) as f:
        for line in f:
            if line.startswith("resolution:"):
                res = float(line.split(":", 1)[1])
            elif line.startswith("origin:"):
                vals = line.split("[", 1)[1].split("]", 1)[0].split(",")
                origin = (float(vals[0]), float(vals[1]))
    with open(pgm_path, "rb") as f:
        assert f.readline().strip() == b"P5", "expected binary PGM (P5)"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        int(f.readline())
        img = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
    extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]
    return np.flipud(img), extent  # flip so origin='lower' matches world y-up


def load_run(results_dir, method, policy, seeds):
    """Per-cell CRPS averaged over seeds; returns {(vx, vy): crps}."""
    acc = {}
    for s in seeds:
        d = np.load(f"{results_dir}/{PREFIX[method]}_{policy}_s{s}/cellmap.npz")
        for vx, vy, c in zip(d["vx"], d["vy"], d["crps"]):
            acc.setdefault((int(vx), int(vy)), []).append(float(c))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="../results/exploration",
                    help="directory holding the per-run cellmap.npz dumps")
    ap.add_argument("--map-pgm", required=True, help="ATC occupancy PGM")
    ap.add_argument("--map-yaml", required=True, help="its .yaml (resolution + origin)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--policy", default="random",
                    help="patrol policy to plot (default: uniform coverage)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--voxel-size", type=float, default=0.4)
    ap.add_argument("--seq-cmap", default=SEQ_CMAP,
                    help="colormap of the absolute maps (default: %(default)s)")
    ap.add_argument("--compare", nargs="*", default=[],
                    metavar="NAME:PREFIX:LABEL:DIFFMAX",
                    help="comparison methods to plot against Kairos; only "
                         "NAME is required")
    args = ap.parse_args()

    for spec in args.compare:
        parts = (spec.split(":") + ["", "", ""])[:4]
        name = parts[0]
        PREFIX[name] = parts[1] or name
        LABEL[name] = parts[2] or name
        if parts[3]:
            DIFF_MAX[name] = float(parts[3])

    VS = args.voxel_size
    os.makedirs(args.out_dir, exist_ok=True)
    runs = {m: load_run(args.results_dir, m, args.policy, args.seeds) for m in PREFIX}

    # shared world extent over every scored cell
    allk = set().union(*[set(a) for a in runs.values()])
    vx = np.array([k[0] for k in allk]); vy = np.array([k[1] for k in allk])
    x0, x1 = vx.min() * VS - 2, (vx.max() + 1) * VS + 2
    y0, y1 = vy.min() * VS - 2, (vy.max() + 1) * VS + 2
    nx, ny = int(round((x1 - x0) / VS)), int(round((y1 - y0) / VS))
    ext = [x0, x1, y0, y1]

    def raster(cell_map):
        g = np.full((ny, nx), np.nan)
        for (a, b), c in cell_map.items():
            ix, iy = int(round((a * VS - x0) / VS)), int(round((b * VS - y0) / VS))
            if 0 <= ix < nx and 0 <= iy < ny:
                g[iy, ix] = c
        return np.ma.masked_invalid(g)

    mp_disp, mext = read_yaml_map(args.map_pgm, args.map_yaml)

    def save(grid, cmap, norm, slug):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        ax.imshow(mp_disp, cmap="gray", extent=mext, origin="lower",
                  vmin=0, vmax=255, alpha=0.35, interpolation="nearest", zorder=0)
        im = ax.imshow(grid, cmap=cmap, norm=norm, extent=ext, origin="lower",
                       interpolation="nearest", zorder=1)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.tick_params(labelsize=8)
        # horizontal bar under the map: stays inside the figure bounding box when
        # the panels are tiled, and needs no rotated label (the caption carries
        # the quantity and its units). Flat ends, so no out-of-range arrows.
        cb = fig.colorbar(im, ax=ax, location="bottom", fraction=0.05,
                          pad=0.02, aspect=40, extend="neither")
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_linewidth(0.6)
        for e in ("png", "pdf"):
            fig.savefig(f"{args.out_dir}/sparse_coverage_{slug}.{e}",
                        dpi=200 if e == "png" else None)
        plt.close(fig)
        print(f"wrote sparse_coverage_{slug}.{{png,pdf}}")

    # three absolute maps on the shared sequential scale
    seq_norm = Normalize(SEQ_LO, SEQ_HI)
    for m in PREFIX:
        save(raster(runs[m]), plt.get_cmap(args.seq_cmap), seq_norm, m)

    # one difference map per comparison method, each on its own diverging scale
    for other in [m for m in PREFIX if m != "kairos"]:
        shared = set(runs["kairos"]) & set(runs[other])
        d = {k: runs["kairos"][k] - runs[other][k] for k in shared}
        mmax = DIFF_MAX.get(other, DIFF_MAX_DEFAULT)
        save(raster(d), plt.cm.RdBu_r, TwoSlopeNorm(vmin=-mmax, vcenter=0.0, vmax=mmax),
             f"diff_kairos_{other}")
        dv = np.fromiter(d.values(), float)
        print(f"  Kairos better than {LABEL[other]} in {(dv < 0).mean() * 100:.0f}%"
              f" of {len(dv)} cells (mean $\\Delta$={dv.mean():+.4f})")


if __name__ == "__main__":
    main()
