#!/usr/bin/env python3
"""Shape of a cell's daily direction rhythm: reversal or dispersion?

measure_periodicity.py tells us WHETHER a cell's direction mix departs from its
all-day average per hour (JS divergence), but not HOW. Two very different
mechanisms give the same JSD signature:
  * reversal  - the dominant direction's share swings across the day (the
                commuter pattern: toward the platforms in the morning, toward
                the exit in the evening);
  * dispersion - the same modes persist all day but their sharpness changes
                (rush-hour crowding widens the heading spread without moving it).
Only the first supports "which way should the robot expect people to walk at
hour h"; both are forecastable structure for the mixing weights.

Per selected cell this prints, for each hour of day:
  movers | share within +-45 deg of the cell's DOMINANT heading | share within
  +-45 deg of the OPPOSITE heading | entropy of the K-bin mix (bits).
A reversal shows as dom/opp shares crossing over; a dispersion change shows as
stable shares with entropy moving.

Cells default to the busiest two-way cells (reverse fraction > 0.2) plus the
busiest overall; pass --cells to inspect specific ids from a
measure_periodicity run (same bounds/voxel => same ids).

Usage:
    python3 inspect_hour_profile.py --source atc --daytype weekday
    python3 inspect_hour_profile.py --source hb --station HB --daytype weekday

Run this after measure_periodicity.py has told you a rhythm exists, to see
which mechanism produces it. Reversal and dispersion give the same divergence
signature but mean different things for forecasting: a reversal is predictable
from the hour, a dispersion is not.
"""
import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import measure_periodicity as mp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=mp.SOURCES,
                    default="hb")
    ap.add_argument("--station", choices=["HB", "PB"], default="HB")
    ap.add_argument("--days", nargs="+", default=None)
    ap.add_argument("--x-bounds", type=float, nargs=2, default=None)
    ap.add_argument("--y-bounds", type=float, nargs=2, default=None)
    ap.add_argument("--voxel-size-m", type=float, default=0.4)
    ap.add_argument("--dir-bins", type=int, default=8)
    ap.add_argument("--hour-bins", type=int, default=24)
    ap.add_argument("--daytype", choices=["all", "weekday", "weekend"], default="weekday")
    ap.add_argument("--min-speed-mps", type=float, default=0.5)
    ap.add_argument("--cells", type=int, nargs="+", default=None,
                    help="cell ids to inspect (default: auto-pick busiest two-way)")
    ap.add_argument("--n-auto", type=int, default=6,
                    help="how many cells to auto-pick when --cells is not given")
    args = ap.parse_args()
    paths = mp.resolve_source(args)

    x0, x1 = args.x_bounds; y0, y1 = args.y_bounds
    vs = args.voxel_size_m; kdir = args.dir_bins; nph = args.hour_bins
    nx = int(math.ceil((x1 - x0) / vs)); ny = int(math.ceil((y1 - y0) / vs))
    ncell = nx * ny
    dtheta = 2.0 * math.pi / kdir
    hist = np.zeros((ncell, nph, kdir))

    print(f"[hour-profile] source={args.source} {len(paths)} files | nphase={nph} "
          f"vox={vs}m K={kdir} min_speed={args.min_speed_mps} daytype={args.daytype}",
          flush=True)
    for path in paths:
        s = mp._stream(args.source, path, tuple(args.x_bounds), tuple(args.y_bounds))
        if len(s.t) == 0:
            continue
        dayno = np.floor(s.t / 86400.0)
        keep = mp.weekday_mask(dayno, args.daytype)
        mv = np.isfinite(s.theta) & keep
        if args.min_speed_mps > 0:
            mv &= np.isfinite(s.rho) & (s.rho > args.min_speed_mps)
        if not mv.any():
            continue
        ix = np.clip(((s.x[mv] - x0) / vs).astype(np.int64), 0, nx - 1)
        iy = np.clip(((s.y[mv] - y0) / vs).astype(np.int64), 0, ny - 1)
        cell = ix * ny + iy
        hod = np.mod(s.t[mv] / 3600.0, 24.0)
        iph = np.clip((hod / (24.0 / nph)).astype(np.int64), 0, nph - 1)
        idir = np.mod(np.floor((s.theta[mv] + math.pi) / dtheta).astype(np.int64), kdir)
        flat = (cell * nph + iph) * kdir + idir
        hist += np.bincount(flat, minlength=ncell * nph * kdir).reshape(ncell, nph, kdir)
        print(f"  [{os.path.basename(path)}] {int(mv.sum()):,} movers", flush=True)

    # Per-hour speed statistics for the same movers: mean and std of rho per
    # (cell, hour). Distinguishes a MEAN shift (capturable by a forecast of the
    # per-slot mean; shows up in Speed MAE) from a SPREAD shift (invisible to a
    # fixed-sigma density and to a mean forecast alike).
    ssum = np.zeros((ncell, nph)); ssq = np.zeros((ncell, nph)); scnt = np.zeros((ncell, nph))
    for path in paths:
        st = mp._stream(args.source, path, tuple(args.x_bounds), tuple(args.y_bounds))
        if len(st.t) == 0:
            continue
        dayno = np.floor(st.t / 86400.0)
        keep = mp.weekday_mask(dayno, args.daytype)
        mv = np.isfinite(st.theta) & keep
        if args.min_speed_mps > 0:
            mv &= np.isfinite(st.rho) & (st.rho > args.min_speed_mps)
        if not mv.any():
            continue
        ix = np.clip(((st.x[mv] - x0) / vs).astype(np.int64), 0, nx - 1)
        iy = np.clip(((st.y[mv] - y0) / vs).astype(np.int64), 0, ny - 1)
        cell = ix * ny + iy
        hod = np.mod(st.t[mv] / 3600.0, 24.0)
        iph = np.clip((hod / (24.0 / nph)).astype(np.int64), 0, nph - 1)
        rho = np.minimum(st.rho[mv], 3.0)  # same physical cap as scoring
        flat = cell * nph + iph
        ssum += np.bincount(flat, weights=rho, minlength=ncell * nph).reshape(ncell, nph)
        ssq += np.bincount(flat, weights=rho * rho, minlength=ncell * nph).reshape(ncell, nph)
        scnt += np.bincount(flat, minlength=ncell * nph).reshape(ncell, nph)

    tot = hist.sum(axis=(1, 2))
    if args.cells:
        cells = [c for c in args.cells if 0 <= c < ncell and tot[c] > 0]
    else:
        # busiest two-way cells (reverse fraction > 0.2), then busiest overall
        cells, extra = [], []
        for c in np.argsort(tot)[::-1]:
            if len(cells) >= args.n_auto:
                break
            m = hist[c].sum(axis=0)
            if m.sum() == 0:
                continue
            dom = int(np.argmax(m))
            fwd = sum(m[(dom + j) % kdir] for j in (-1, 0, 1))
            opp = dom + kdir // 2
            bwd = sum(m[(opp + j) % kdir] for j in (-1, 0, 1))
            rev = bwd / (fwd + bwd) if (fwd + bwd) > 0 else 0.0
            (cells if rev > 0.2 else extra).append(int(c))
        cells += extra[: max(0, args.n_auto - len(cells))]

    for c in cells:
        h = hist[c]                                        # (nph, kdir)
        m = h.sum(axis=0)
        dom = int(np.argmax(m))
        opp = (dom + kdir // 2) % kdir
        deg = (dom + 0.5) * 360.0 / kdir - 180.0
        gx, gy = c // ny, c % ny
        xm, ym = x0 + (gx + 0.5) * vs, y0 + (gy + 0.5) * vs
        n_ph = h.sum(axis=1)
        share = np.full(nph, np.nan); oshare = np.full(nph, np.nan)
        ent = np.full(nph, np.nan)
        for p in range(nph):
            if n_ph[p] <= 0:
                continue
            hp = h[p] / n_ph[p]
            share[p] = sum(hp[(dom + j) % kdir] for j in (-1, 0, 1))
            oshare[p] = sum(hp[(opp + j) % kdir] for j in (-1, 0, 1))
            nzero = hp[hp > 0]
            ent[p] = float(-(nzero * np.log2(nzero)).sum())
        hrs = 24.0 / nph
        print(f"\ncell {c} (x~{xm:+.1f}m y~{ym:+.1f}m) {int(tot[c]):,} movers, "
              f"dominant {deg:+.0f} deg")
        print("  hour  : " + " ".join(f"{p * hrs:5.1f}" for p in range(nph)))
        print("  movers: " + " ".join(f"{int(v):5d}" for v in n_ph))
        print("  dom%  : " + " ".join("    -" if np.isnan(v) else f"{100 * v:5.0f}"
                                      for v in share))
        print("  opp%  : " + " ".join("    -" if np.isnan(v) else f"{100 * v:5.0f}"
                                      for v in oshare))
        print("  ent_b : " + " ".join("    -" if np.isnan(v) else f"{v:5.2f}"
                                      for v in ent))
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = ssum[c] / scnt[c]
            sd = np.sqrt(np.maximum(ssq[c] / scnt[c] - mu * mu, 0.0))
        print("  v_mu  : " + " ".join("    -" if not np.isfinite(v) else f"{v:5.2f}"
                                      for v in mu))
        print("  v_sd  : " + " ".join("    -" if not np.isfinite(v) else f"{v:5.2f}"
                                      for v in sd))
    print("\nRead: dom%/opp% crossing or swinging across hours = reversal (the mix "
          "itself is time-dependent); stable shares with entropy moving = "
          "dispersion change only. v_mu moving = speed MEAN shift (a mean "
          "forecast can capture it; ceiling on the Speed-MAE gain = the swing); "
          "v_mu flat with v_sd moving = SPREAD shift (invisible to a fixed-sigma "
          "density).")


if __name__ == "__main__":
    main()
