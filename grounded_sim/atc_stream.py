#!/usr/bin/env python3
"""ATC pedestrian detection stream.

Loads an ATC tracking CSV once into sorted numpy arrays and serves the
detections in a time window, in world metres with (heading, speed). Columns:
  time[unix s], id, x[mm], y[mm], z[mm], velocity[mm/s], angle_of_motion[rad],
  facing_angle[rad]
heading = angle_of_motion; speed = velocity/1000.
"""
import numpy as np
# pandas is imported lazily inside the CSV path only: the compact .npz path
# (the normal case) needs only numpy, which keeps the pipeline container's
# buildless runtime dependency set minimal.

COLS = ["t", "id", "x", "y", "z", "vel", "ang", "face"]


def load_occupancy(pgm_path, yaml_path, vs):
    """Load the given ROS occupancy grid, resampled to the vs-metre voxel grid.

    Returns (free, occ, x0, y0, nx, ny) where `free`/`occ` are (nx, ny) boolean
    masks on the same voxel grid the flow voxels use (origin x0,y0, cell vs),
    so the fixed navigational layer built from `free` overlays the incremental
    voxels exactly. This grid is effectively binary (mid-grey navigable vs black
    walls), so we split at 60: >=60 free, <60 wall. Image row 0 is the top
    (max y), so y is flipped when sampling."""
    from PIL import Image
    res = origin = None
    with open(yaml_path) as f:
        for line in f:
            if line.startswith("resolution:"):
                res = float(line.split(":")[1])
            if line.startswith("origin:"):
                origin = [float(v) for v in line.split("[")[1].split("]")[0].split(",")]
    img = np.array(Image.open(pgm_path))
    if img.ndim == 3:
        img = img[..., 0]
    H, W = img.shape
    ox, oy = origin[0], origin[1]
    x1, y1 = ox + W * res, oy + H * res
    nx = int(np.ceil((x1 - ox) / vs))
    ny = int(np.ceil((y1 - oy) / vs))
    cx = ox + (np.arange(nx) + 0.5) * vs
    cy = oy + (np.arange(ny) + 0.5) * vs
    col = np.clip(((cx - ox) / res).astype(int), 0, W - 1)          # per x-index
    row = np.clip((H - 1 - ((cy - oy) / res)).astype(int), 0, H - 1)  # per y-index (flip)
    samp = img[np.ix_(row, col)].T                                  # (nx, ny)
    free = samp >= 60
    occ = samp < 60
    return free, occ, ox, oy, nx, ny


def map_bounds(pgm_path, yaml_path):
    """The occupancy map's extent as ((x0, x1), (y0, y1)) in world metres: the
    canonical detection rectangle every ATC scorer shares (detections beyond it
    lie outside the mapped environment that all method substrates are built
    on). Same parse as load_occupancy."""
    from PIL import Image
    res = origin = None
    with open(yaml_path) as f:
        for line in f:
            if line.startswith("resolution:"):
                res = float(line.split(":")[1])
            if line.startswith("origin:"):
                origin = [float(v) for v in line.split("[")[1].split("]")[0].split(",")]
    img = np.array(Image.open(pgm_path))
    if img.ndim == 3:
        img = img[..., 0]
    H, W = img.shape
    return (origin[0], origin[0] + W * res), (origin[1], origin[1] + H * res)


class AtcStream:
    def __init__(self, path, x_bounds=None, y_bounds=None):
        """Load a single ATC day. `path` may be either a raw `atc-*.csv` or the
        compact `atc-*.npz` produced by extract_days.py (columns t,id,x,y,vel,ang
        in the CSV's raw units: unix s, mm, mm/s, rad). Either way the stream is
        served in world metres with (heading, speed)."""
        if str(path).endswith(".npz"):
            z = np.load(path)
            t = z["t"].astype(np.float64)
            iid = z["id"]; xr = z["x"]; yr = z["y"]; ang = z["ang"]; vel = z["vel"]
        else:
            import pandas as pd
            df = pd.read_csv(path, header=None, names=COLS,
                             usecols=["t", "id", "x", "y", "vel", "ang"],
                             dtype={"t": np.float64, "id": np.int64, "x": np.float64,
                                    "y": np.float64, "vel": np.float64, "ang": np.float64})
            t = df.t.values; iid = df.id.values
            xr = df.x.values; yr = df.y.values; ang = df.ang.values; vel = df.vel.values
        order = np.argsort(t, kind="stable")
        self.t = t[order]
        self.id = iid[order]                           # person id (for tracks)
        self.x = xr[order] / 1000.0                    # mm -> m
        self.y = yr[order] / 1000.0
        self.theta = ang[order]                        # rad
        self.rho = vel[order] / 1000.0                 # mm/s -> m/s
        if x_bounds and y_bounds:
            m = ((self.x >= x_bounds[0]) & (self.x < x_bounds[1]) &
                 (self.y >= y_bounds[0]) & (self.y < y_bounds[1]))
            self.t, self.id, self.x, self.y, self.theta, self.rho = (
                self.t[m], self.id[m], self.x[m], self.y[m], self.theta[m], self.rho[m])
        self.t_min, self.t_max = float(self.t[0]), float(self.t[-1])

    def between(self, t0, t1):
        """Detections with timestamp in [t0, t1). Returns (x, y, theta, rho, id)."""
        lo = np.searchsorted(self.t, t0, "left")
        hi = np.searchsorted(self.t, t1, "left")
        s = slice(lo, hi)
        return self.x[s], self.y[s], self.theta[s], self.rho[s], self.id[s]
