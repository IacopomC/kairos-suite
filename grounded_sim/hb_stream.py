#!/usr/bin/env python3
"""Havlickuv Brod (HB) train-station concourse stream (Apeltauer et al. 2024,
Sci Data).

One CSV per camera per day: columns id, relative_time[s], datetime, x[m], y[m].
Column 3 is a full wall-clock datetime, so the real date and time-of-day are known
(unlike Eindhoven, whose clock was stripped). Cameras within a station share one
local metric frame, so per-camera files for a station just concatenate. Sampling is
~1 Hz; no heading/speed is stored, so both are derived from per-track displacement
over a short window (default 3 s).

Exposes the AtcStream interface: t (s, absolute; t mod 86400 = local time-of-day,
floor(t/86400) increments per calendar day and %7 gives a real weekday), id,
x/y (m), theta (rad), rho (m/s).
"""
import numpy as np
import pandas as pd


class HBStream:
    def __init__(self, path, x_bounds=None, y_bounds=None, win=3):
        df = pd.read_csv(path, skiprows=1, header=None,
                         names=["id", "rt", "dt", "x", "y"])
        # Naive local datetime -> seconds; treating it as UTC means t mod 86400 is
        # local seconds-into-day and floor(t/86400)%7 is a consistent real weekday.
        # Force second resolution explicitly: pandas>=2 datetime unit varies, so
        # .astype('int64') is NOT reliably nanoseconds. datetime64[s]->int64 is
        # seconds-since-epoch regardless of the source unit.
        t = (pd.to_datetime(df["dt"], format="%Y-%m-%d %H:%M:%S")
             .to_numpy().astype("datetime64[s]").astype(np.int64).astype(np.float64))
        iid = df["id"].to_numpy()
        x = df["x"].to_numpy(dtype=np.float64)
        y = df["y"].to_numpy(dtype=np.float64)

        order = np.lexsort((t, iid))              # sort by track, then time
        t = t[order]; iid = iid[order]; x = x[order]; y = y[order]

        theta = np.full(t.size, np.nan); rho = np.full(t.size, np.nan)
        if t.size > win:
            dx = x[win:] - x[:-win]; dy = y[win:] - y[:-win]; dt = t[win:] - t[:-win]
            same = (iid[win:] == iid[:-win]) & (dt > 0)
            idx = np.nonzero(same)[0]
            theta[idx] = np.arctan2(dy[idx], dx[idx])
            rho[idx] = np.sqrt(dx[idx] ** 2 + dy[idx] ** 2) / dt[idx]

        self.t = t; self.id = iid; self.x = x; self.y = y
        self.theta = theta; self.rho = rho
        if x_bounds and y_bounds:
            m = ((self.x >= x_bounds[0]) & (self.x < x_bounds[1]) &
                 (self.y >= y_bounds[0]) & (self.y < y_bounds[1]))
            self.t = self.t[m]; self.id = self.id[m]; self.x = self.x[m]
            self.y = self.y[m]; self.theta = self.theta[m]; self.rho = self.rho[m]


import glob
import os
from datetime import date as _date


def hb_day_files(root, station, date_iso):
    """The station's per-camera files for one date (may be fewer than all
    cameras; callers that need constant coverage should use station_dates)."""
    return sorted(glob.glob(os.path.join(root, f"{station}-CAM*",
                                         f"*_{date_iso}_trans.csv")))


def station_dates(root, station):
    """Dates covered by EVERY camera of the station -> {date_iso: [files]}.
    Restricting to all-camera dates keeps the spatial coverage of the
    concatenated stream constant across the span."""
    cams = sorted(glob.glob(os.path.join(root, f"{station}-CAM*")))
    per_cam = []
    for cam in cams:
        d = {}
        for f in glob.glob(os.path.join(cam, "*_trans.csv")):
            d[os.path.basename(f).split("_")[1]] = f
        per_cam.append(d)
    common = set(per_cam[0])
    for d in per_cam[1:]:
        common &= set(d)
    return {dt: [d[dt] for d in per_cam] for dt in sorted(common)}


def _load_raw(path):
    """One camera file -> (t seconds-since-epoch float64, id, x, y), unsorted."""
    df = pd.read_csv(path, skiprows=1, header=None,
                     names=["id", "rt", "dt", "x", "y"])
    t = (pd.to_datetime(df["dt"], format="%Y-%m-%d %H:%M:%S")
         .to_numpy().astype("datetime64[s]").astype(np.int64).astype(np.float64))
    return t, df["id"].to_numpy(np.int64), df["x"].to_numpy(np.float64), \
        df["y"].to_numpy(np.float64)


class HBDayStream:
    """One station-date served at an ATC-like effective frame rate.

    Concatenates the per-camera files (track ids are globally unique across
    cameras) and linearly interpolates each track to `interp_hz` between
    consecutive samples closer than `gap_max_s`, so a voxel crossing leaves
    detections at the same density as a 10 fps stream and the windowed
    occupancy event keeps its person-level meaning. Heading and speed are the
    segment displacement direction and rate; a sample followed by a gap
    larger than `gap_max_s` (or a track end) is kept as a lone point with the
    preceding segment's heading/speed (0/0 for single-sample tracks).
    Time-sorted; exposes the AtcStream interface."""

    def __init__(self, files, x_bounds=None, y_bounds=None, interp_hz=10,
                 gap_max_s=2.0):
        ts, ids, xs, ys = [], [], [], []
        for f in files:
            t, i, x, y = _load_raw(f)
            ts.append(t); ids.append(i); xs.append(x); ys.append(y)
        t = np.concatenate(ts); iid = np.concatenate(ids)
        x = np.concatenate(xs); y = np.concatenate(ys)
        if x_bounds and y_bounds:
            m = ((x >= x_bounds[0]) & (x < x_bounds[1]) &
                 (y >= y_bounds[0]) & (y < y_bounds[1]))
            t, iid, x, y = t[m], iid[m], x[m], y[m]
        order = np.lexsort((t, iid))
        t, iid, x, y = t[order], iid[order], x[order], y[order]

        hz = float(interp_hz)
        # Segments between consecutive samples of the same track.
        same = iid[1:] == iid[:-1]
        dt = t[1:] - t[:-1]
        seg = same & (dt > 0) & (dt <= gap_max_s)
        s_dt = dt[seg]
        s_dx = (x[1:] - x[:-1])[seg]; s_dy = (y[1:] - y[:-1])[seg]
        s_t0 = t[:-1][seg]; s_x0 = x[:-1][seg]; s_y0 = y[:-1][seg]
        s_id = iid[:-1][seg]
        s_th = np.arctan2(s_dy, s_dx)
        s_rho = np.hypot(s_dx, s_dy) / s_dt
        # n points per segment at 1/hz spacing, t0 included, t1 excluded (the
        # next segment or the lone endpoint supplies it).
        n = np.maximum(np.rint(s_dt * hz).astype(np.int64), 1)
        tot = int(n.sum())
        rep = np.repeat(np.arange(n.size), n)
        offs = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
        ti = s_t0[rep] + offs / hz
        frac = (offs / hz) / s_dt[rep]
        xi = s_x0[rep] + frac * s_dx[rep]
        yi = s_y0[rep] + frac * s_dy[rep]
        thi = s_th[rep]; rhi = s_rho[rep]; idi = s_id[rep]

        # Lone points: rows not serving as a segment start (track end, or a
        # gap larger than gap_max ahead), with the previous segment's values.
        start = np.append(seg, False)                 # row i starts a segment
        lone = ~start
        prev_seg = np.append(False, seg)              # row i ends a segment
        th_prev = np.zeros(t.size); rho_prev = np.zeros(t.size)
        th_prev[1:][seg] = s_th; rho_prev[1:][seg] = s_rho
        keep = lone
        tl, xl, yl, il = t[keep], x[keep], y[keep], iid[keep]
        thl = np.where(prev_seg[keep], th_prev[keep], 0.0)
        rhl = np.where(prev_seg[keep], rho_prev[keep], 0.0)

        T = np.concatenate([ti, tl]); X = np.concatenate([xi, xl])
        Y = np.concatenate([yi, yl]); TH = np.concatenate([thi, thl])
        RH = np.concatenate([rhi, rhl]); ID = np.concatenate([idi, il])
        o = np.argsort(T, kind="stable")
        self.t, self.x, self.y = T[o], X[o], Y[o]
        self.theta, self.rho, self.id = TH[o], RH[o], ID[o]
        self.t_min, self.t_max = float(self.t[0]), float(self.t[-1])

    def between(self, t0, t1):
        """Detections with timestamp in [t0, t1). Returns (x, y, theta, rho, id)."""
        lo = np.searchsorted(self.t, t0, "left")
        hi = np.searchsorted(self.t, t1, "left")
        s = slice(lo, hi)
        return self.x[s], self.y[s], self.theta[s], self.rho[s], self.id[s]
