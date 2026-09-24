#!/usr/bin/env python3
"""Edinburgh Informatics Forum pedestrian stream (one day's tracks file).

Format: one text file per day, `tracks.DDMon.txt`. Each trajectory is
  TRACK.R<n>=[[x y T]; [x y T]; ...];
with x,y in pixels (24.7 mm/px) and T the time since file start in 0.1 s units
(so it is real elapsed time regardless of the ~9 fps variability). Recordings
start ~08:00; the blobs files that carried the exact BEGIN unix stamp are not
present, so absolute time-of-day is approximate: we anchor each day at 08:00 local
using the date parsed from the filename (year inferred from the Aug 2009 - Aug 2010
collection window). That is fine for time-of-day folds at ~minutes resolution but
carries a small per-day start jitter; the real weekday is exact from the date.

Exposes the AtcStream interface: t (s, absolute-ish), id, x/y (m), theta, rho.
"""
import datetime
import os
import re

import numpy as np

_PX_TO_M = 0.0247
_TRIP = re.compile(r"\[(\d+)\s+(\d+)\s+(\d+)\]")
_MON = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _day_base_unix(path):
    """Midnight-unix of the day from the filename, +8h opening anchor.
    Collection ran Aug 2009 - Aug 2010, so Aug-Dec -> 2009, Jan-Jul -> 2010."""
    tok = os.path.basename(path).replace("tracks.", "").replace(".txt", "")
    dd = int(tok[:2]); mon = _MON[tok[2:5]]
    year = 2009 if mon >= 8 else 2010
    days = (datetime.date(year, mon, dd) - datetime.date(1970, 1, 1)).days
    return days * 86400 + 8 * 3600            # +8h: recordings open ~08:00


class EdinburghStream:
    def __init__(self, path, x_bounds=None, y_bounds=None, win=10):
        base = _day_base_unix(path)
        ts, ids, xs, ys, ths, rhos = [], [], [], [], [], []
        tid = 0
        for ch in open(path).read().split("TRACK.R")[1:]:
            tr = _TRIP.findall(ch)
            if len(tr) <= win:
                continue
            a = np.array(tr, dtype=np.float64)          # (n,3): px_x, px_y, T
            x = a[:, 0] * _PX_TO_M; y = a[:, 1] * _PX_TO_M
            t = base + a[:, 2] * 0.1                     # 0.1 s units -> s
            th = np.full(x.size, np.nan); rh = np.full(x.size, np.nan)
            dx = x[win:] - x[:-win]; dy = y[win:] - y[:-win]; dt = t[win:] - t[:-win]
            ok = dt > 0
            idx = np.nonzero(ok)[0]
            th[idx] = np.arctan2(dy[idx], dx[idx])
            rh[idx] = np.sqrt(dx[idx] ** 2 + dy[idx] ** 2) / dt[idx]
            ts.append(t); xs.append(x); ys.append(y); ths.append(th); rhos.append(rh)
            ids.append(np.full(x.size, tid)); tid += 1
        if ts:
            self.t = np.concatenate(ts); self.id = np.concatenate(ids)
            self.x = np.concatenate(xs); self.y = np.concatenate(ys)
            self.theta = np.concatenate(ths); self.rho = np.concatenate(rhos)
        else:
            self.t = self.id = self.x = self.y = self.theta = self.rho = np.array([])
        if x_bounds and y_bounds and self.t.size:
            m = ((self.x >= x_bounds[0]) & (self.x < x_bounds[1]) &
                 (self.y >= y_bounds[0]) & (self.y < y_bounds[1]))
            self.t = self.t[m]; self.id = self.id[m]; self.x = self.x[m]
            self.y = self.y[m]; self.theta = self.theta[m]; self.rho = self.rho[m]
