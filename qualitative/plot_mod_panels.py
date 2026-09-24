#!/usr/bin/env python3
r"""Draw a per-cell mixture over the dataset's floor plan, one panel per hour.

Reads the files ``export_flow_mod.py`` writes (one row per mixture component,
with the component's direction, the cell's speed and the component's weight)
and draws two panels per file:

  circular_mean    one arrow per cell, the weight-weighted circular mean
                   direction at a fixed length. Drawing the heaviest component
                   instead would quantise the direction to the representation's
                   fixed component centres.
  all_components   every component from the cell centre, length the predicted
                   speed, shaft thinning and fading as the weight drops. Shows
                   the multimodality the mean panel hides.

Colour is the heading, one hue per direction, so the same direction reads the
same in every panel. The background is a ROS occupancy grid, whose yaml fixes
the world metres the image spans, so the arrows and the plan share one frame
with no alignment step.

Needs numpy, matplotlib and pillow, not the pipeline bindings, so it runs
outside the container.

Usage:
    python3 plot_mod_panels.py --mods /out/qualitative/open \
        --map-yaml /data/atc/ATC-map/localization_grid.yaml \
        --crop -20,0,-10,10 --out-dir /out/qualitative/panels_open
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image

NORM, CMAP = Normalize(0, 360), cm.hsv


def load_map(yaml_path, wall_below=60):
    """The occupancy grid as a plottable image plus the metres it spans.

    Returns (image, extent). Walls are dark on white: the grid is effectively
    binary (mid-grey navigable against black wall), so it is thresholded rather
    than shown raw, which would tint the whole panel.
    """
    res = image = None
    origin = []
    in_origin = False
    with open(yaml_path) as f:
        for line in f:
            if line.startswith("resolution:"):
                res = float(line.split(":")[1])
            elif line.startswith("image:"):
                image = line.split(":", 1)[1].strip()
            elif line.startswith("origin:"):
                rest = line.split(":", 1)[1].strip()
                if rest.startswith("["):          # inline: origin: [x, y, yaw]
                    origin = [float(v) for v in rest.strip("[]").split(",")]
                else:                             # block: one "- value" per line
                    in_origin = True
            elif in_origin and line.startswith("- "):
                origin.append(float(line[2:]))
            elif in_origin:
                in_origin = False
    if len(origin) < 2 or res is None or image is None:
        raise SystemExit("{}: no image, origin or resolution".format(yaml_path))
    raw = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), image)))
    if raw.ndim == 3:
        raw = raw[..., 0]
    img = np.where(raw < wall_below, 60, 255).astype(np.uint8)
    h, w = img.shape
    extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]
    return img, extent


def load_mod(path):
    """Mixture components as rows of (x, y, speed, angle, weight), the rows of
    one cell adjacent."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    a = np.c_[d["x"], d["y"], d["mean_speed"], d["mean_motion_angle"], d["weight"]]
    a[:, 3] = np.mod(a[:, 3], 2 * np.pi)
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def cells(m):
    """The rows of one cell at a time."""
    key = np.round(m[:, 0] * 1e5 + m[:, 1], 3)
    _, start, count = np.unique(key, return_index=True, return_counts=True)
    for s, c in zip(start, count):
        yield m[s:s + c]


def frame(bg, extent, crop):
    plt.clf(); plt.close("all")
    plt.figure(figsize=(5.5, 5.5) if crop else (10, 6), dpi=170)
    plt.rcParams["pdf.fonttype"] = 42
    plt.imshow(bg, cmap="gray", vmin=0, vmax=255, extent=extent)


def finish(out, title, extent, crop, axes=False):
    """Close a panel. The frame carries no axes and no title by default: a
    panel is identified by its file name, and the metres of the crop are the
    caption's business, not the figure's."""
    view = crop if crop else extent
    plt.xlim(view[0], view[1]); plt.ylim(view[2], view[3])
    plt.gca().set_aspect("equal")
    if not axes:
        plt.axis("off")
    if title:
        plt.title(title, fontsize=9)
    plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close("all")


def circular_mean(m, out, bg, extent, crop=None, title=None, axes=False):
    frame(bg, extent, crop)
    length = 0.75 if crop else 0.9
    X, Y, U, V, C = [], [], [], [], []
    for b in cells(m):
        w = b[:, 4] / b[:, 4].sum()
        a = np.arctan2((w * np.sin(b[:, 3])).sum(), (w * np.cos(b[:, 3])).sum())
        X.append(b[0, 0]); Y.append(b[0, 1])
        U.append(length * np.cos(a)); V.append(length * np.sin(a))
        C.append(CMAP(NORM(np.degrees(a % (2 * np.pi)))))
    plt.quiver(X, Y, U, V, color=C, angles="xy", scale_units="xy", scale=1,
               width=0.004, headwidth=3.2, headlength=3.6, headaxislength=3.0)
    finish(out, title, extent, crop, axes)
    return len(X)


def all_components(m, out, bg, extent, crop=None, title=None,
                   min_weight=0.05, nbins=5, weight_length=False, axes=False):
    """Weights are relative to the heaviest component of the cell; those under
    `min_weight` are dropped. With `weight_length` the arrow is shortened by
    its weight as well, which is what keeps a mixture spread over many
    directions readable. Drawn in weight bins because one quiver call carries
    one shaft width."""
    frame(bg, extent, crop)
    rows = []
    for b in cells(m):
        w = b[:, 4] / b[:, 4].max()
        for i in range(len(b)):
            if w[i] >= min_weight:
                rows.append((b[0, 0], b[0, 1], b[i, 2], b[i, 3], w[i]))
    r = np.array(rows)
    edges = np.linspace(min_weight, 1.0, nbins + 1)
    for k in range(nbins):
        sel = (r[:, 4] >= edges[k]) & (r[:, 4] <= edges[k + 1])
        if not sel.any():
            continue
        q = r[sel]
        f = (k + 1) / nbins
        col = CMAP(NORM(np.degrees(q[:, 3])))
        col[:, 3] = 0.20 + 0.80 * f
        length = q[:, 2] * q[:, 4] if weight_length else q[:, 2]
        plt.quiver(q[:, 0], q[:, 1],
                   length * np.cos(q[:, 3]), length * np.sin(q[:, 3]),
                   color=col, angles="xy", scale_units="xy", scale=1,
                   width=0.0012 + 0.0040 * f, headwidth=3, headlength=5,
                   headaxislength=4.5)
    finish(out, title, extent, crop, axes)
    return len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", required=True,
                    help="directory of per-hour files, or one file")
    ap.add_argument("--map-yaml", required=True, help="occupancy grid yaml")
    ap.add_argument("--crop", default=None, help="x0,x1,y0,y1 to draw")
    ap.add_argument("--label", default="",
                    help="name written above each panel, empty for none")
    ap.add_argument("--min-weight", type=float, default=0.05,
                    help="drop a component under this share of its cell's "
                         "heaviest one")
    ap.add_argument("--weight-length", action="store_true",
                    help="shorten each component by its weight as well as "
                         "drawing it thinner")
    ap.add_argument("--axes", action="store_true",
                    help="keep the axes and their metre ticks, for reading a "
                         "panel rather than publishing it")
    ap.add_argument("--minute", type=int, default=30,
                    help="minute past the hour the titles name")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    crop = [float(v) for v in a.crop.split(",")] if a.crop else None
    bg, extent = load_map(a.map_yaml)
    paths = ([a.mods] if os.path.isfile(a.mods)
             else sorted(glob.glob(os.path.join(a.mods, "*.csv"))))
    if not paths:
        raise SystemExit("no map files under {}".format(a.mods))
    if not os.path.isdir(a.out_dir):
        os.makedirs(a.out_dir)
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        m = load_mod(path)
        for panel, draw in (("circular_mean", circular_mean),
                            ("all_components", all_components)):
            title = ("{} {}:{:02d} ({})".format(a.label, stem, a.minute, panel)
                     if a.label else None)
            out = os.path.join(a.out_dir, "{}_{}.png".format(panel, stem))
            if panel == "all_components":
                n = draw(m, out, bg, extent, crop, title,
                         min_weight=a.min_weight, weight_length=a.weight_length,
                         axes=a.axes)
            else:
                n = draw(m, out, bg, extent, crop, title, axes=a.axes)
            print("[panels] {} <- {} arrows".format(out, n), flush=True)


if __name__ == "__main__":
    main()
