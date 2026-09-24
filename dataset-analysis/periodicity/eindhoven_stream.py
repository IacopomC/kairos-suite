#!/usr/bin/env python3
"""Eindhoven Centraal platform pedestrian stream (Zenodo 13784588).

One parquet file = 10 consecutive days of a single continuous 60-day recording at
10 fps. Columns: time_ms (elapsed ms from ~midnight of day 0; one continuous clock
across all six files), object_identifier (track id), x_position_mm, y_position_mm.

No heading/speed is stored, so both are derived per track from displacement over a
short window (default 0.5 s / 5 frames) to beat 10 fps position noise. Heading is
undefined for the last `win` frames of each track (~1% of rows) -> theta/rho NaN;
those rows keep a valid position so they still count toward occupancy.

Exposes the same interface as AtcStream: t (s, elapsed), id, x/y (m), theta (rad),
rho (m/s). t is elapsed-from-start, not unix, but t mod 86400 gives seconds into
the day (start ~midnight) and floor(t/86400) the day index, which is all the
periodicity fold needs.
"""
import numpy as np
import pyarrow.parquet as pq


class EindhovenStream:
    def __init__(self, path, x_bounds=None, y_bounds=None, win=5):
        tbl = pq.read_table(path, columns=["time_ms", "object_identifier",
                                           "x_position_mm", "y_position_mm"])
        t = tbl["time_ms"].to_numpy().astype(np.float64) / 1000.0     # s
        iid = tbl["object_identifier"].to_numpy()
        x = tbl["x_position_mm"].to_numpy().astype(np.float64) / 1000.0  # m
        y = tbl["y_position_mm"].to_numpy().astype(np.float64) / 1000.0

        # Sort by (track, time) so displacement is within-track and monotone.
        order = np.lexsort((t, iid))
        t = t[order]; iid = iid[order]; x = x[order]; y = y[order]

        theta = np.full(t.size, np.nan)
        rho = np.full(t.size, np.nan)
        if t.size > win:
            dx = x[win:] - x[:-win]
            dy = y[win:] - y[:-win]
            dt = t[win:] - t[:-win]
            same = (iid[win:] == iid[:-win]) & (dt > 0)               # same track
            idx = np.nonzero(same)[0]                                 # start rows
            theta[idx] = np.arctan2(dy[idx], dx[idx])
            rho[idx] = np.sqrt(dx[idx] ** 2 + dy[idx] ** 2) / dt[idx]

        self.t = t
        self.id = iid
        self.x = x
        self.y = y
        self.theta = theta
        self.rho = rho
        if x_bounds and y_bounds:
            m = ((self.x >= x_bounds[0]) & (self.x < x_bounds[1]) &
                 (self.y >= y_bounds[0]) & (self.y < y_bounds[1]))
            self.t = self.t[m]; self.id = self.id[m]; self.x = self.x[m]
            self.y = self.y[m]; self.theta = self.theta[m]; self.rho = self.rho[m]
