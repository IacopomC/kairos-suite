#!/usr/bin/env python3
r"""Model-free periodicity measurement on the raw god's-eye ATC stream (fixes 3.1).

Question: does the DIRECTION MIX have a daily (hour-of-day) rhythm at all,
independent of any model? And, as a positive control, does OCCUPANCY (how many)?
This decides the forecasting narrative before any estimator change is made.

Method (per cell, god's-eye = every detection, no FOV limit):
  * Voxelise (x,y) at the flow voxel size; keep the top-N busiest cells.
  * Direction channel: per hour-of-day h, pool the K-bin heading histogram over
    training days -> P[h]. The rhythm signal is how far each hour departs from the
    cell's all-hours marginal Pbar: S = sum_h w_h * JS(P[h] || Pbar).
  * Occupancy channel: per hour-of-day, the mean detection count -> a 24-vector
    profile; signal = JS(profile || uniform), i.e. non-flatness of the daily curve.

Noise floor (the point of this design): accumulate EVEN-day and ODD-day
histograms separately in ONE pass. "Same hour, different days" = JS(P_even[h] ||
P_odd[h]) is a split-half sampling-noise estimate. A rhythm is real only when the
across-hour signal exceeds this within-hour across-day noise:
      rhythm index  =  S_across / N_within   (>> 1 means real periodicity).

By default every detection votes. With --dedup the direction channel counts one
vote per (track, cell, phase) instead of one per frame: at 10 fps a single
crossing yields tens of identical votes and a loiterer thousands, which inflates
the split-half noise floor far above true sampling noise (the effective sample
is passages, not frames). Occupancy stays per-frame either way, since presence
is person-time by definition.

Voxel size, direction-bin count, hour count and top-N are CLI params (a dataset's
recurrence scale is never hardcoded).

Usage:
    # Does ATC's direction mix have a daily rhythm? (the headline read)
    python3 measure_periodicity.py --source atc --fold hour --hour-bins 24 \
        --daytype weekday --min-speed-mps 0.5 --dedup

    # A station concourse, same question
    python3 measure_periodicity.py --source hb --station HB --fold hour --dedup

    # A short recurrence instead of a daily one (a train cadence, say)
    python3 measure_periodicity.py --source eindhoven --fold period --period-min 30

Read the rhythm index: much greater than 1 means the across-hour signal beats
the split-half noise floor, so the periodicity is real. Near 1 means what looks
like a rhythm is sampling noise.

Prefer --dedup for the direction channel: at 10 fps one crossing otherwise
casts tens of identical votes and a loiterer thousands, which inflates the
noise floor above true sampling noise.

Adding a dataset: write a loader module taking (path, x_bounds, y_bounds) and
add one row to the DATASETS table. Nothing else dispatches on the name.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets"))

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# The suite's own stream modules (atc, hb, tbd) live alongside the trainer.
sys.path.insert(0, os.path.join(_HERE, "..", "..", "grounded_sim"))

TEST_DAYS = ["20130901", "20130904", "20130911", "20130915",
             "20130918", "20130922", "20130925", "20130929"]


# ---------------------------------------------------------------- datasets
# One row per dataset, holding everything that varies between them:
#
#   loader  the stream class, imported lazily so a missing optional dependency
#           only breaks the dataset that needs it
#   bounds  default (x_bounds, y_bounds) map extent, used when the caller gives
#           none. Cell ids are derived from these, so two scripts must resolve
#           the same bounds for their ids to be comparable.
#   files   callable(args) -> the day-file list, when --days is not given
#
# Adding a dataset: write a loader module taking (path, x_bounds, y_bounds),
# then add one row here. Nothing else dispatches on the dataset name.


def _atc_files(args):
    npz_dir = getattr(args, "npz_dir", None) or os.path.join(DATA_ROOT, "atc/npz")
    days = sorted(glob.glob(os.path.join(npz_dir, "atc-*.npz")))
    return [d for d in days if os.path.basename(d)[4:12] not in TEST_DAYS]


def _eindhoven_files(args):
    data_dir = getattr(args, "data_dir", None) or os.path.join(DATA_ROOT, "eindhoven-station")
    return sorted(glob.glob(os.path.join(data_dir, "*.parquet")))


def _tbd_files(args):
    return sorted(glob.glob(os.path.join(DATA_ROOT, "tbd", "*", "scene_tbd_*", "tracks.csv")))


def _hb_files(args):
    root = os.path.join(DATA_ROOT, "hb-station/Data-2")
    return sorted(glob.glob(os.path.join(root, f"{args.station}-CAM*", "*_trans.csv")))


def _edinburgh_files(args):
    return sorted(glob.glob(os.path.join(DATA_ROOT, "edinburgh-forum", "tracks", "tracks.*.txt")))


def _hb_bounds(args):
    """HB's two stations have different concourse extents."""
    return (((-2.0, 35.0), (-32.0, 2.0)) if args.station == "HB"
            else ((-4.0, 82.0), (-22.0, 4.0)))


DATASETS = {
    "atc": dict(loader=("atc_stream", "AtcStream"),
                bounds=((-45.0, 50.0), (-30.0, 25.0)), files=_atc_files),
    "hb": dict(loader=("hb_stream", "HBStream"),
               bounds=_hb_bounds, files=_hb_files),
    "tbd": dict(loader=("tbd_stream", "TbdStream"),
                bounds=((-5.0, 25.0), (-25.0, 0.0)), files=_tbd_files),
    "eindhoven": dict(loader=("eindhoven_stream", "EindhovenStream"),
                      bounds=((-10.0, 80.0), (-12.0, 10.0)), files=_eindhoven_files),
    "edinburgh": dict(loader=("edinburgh_stream", "EdinburghStream"),
                      bounds=((-1.0, 17.0), (-1.0, 13.0)), files=_edinburgh_files),
}
SOURCES = sorted(DATASETS)


def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence (base 2) between two histograms / profiles.
    Inputs need not be normalised; zero-sum rows return 0."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 0.0
    p = p / sp + eps; q = q / sq + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * (np.log2(a) - np.log2(b))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _stream(source, path, xb, yb):
    """Open one day of `source` clipped to the given bounds."""
    import importlib
    module, cls = DATASETS[source]["loader"]
    return getattr(importlib.import_module(module), cls)(
        path, x_bounds=xb, y_bounds=yb)


def weekday_mask(dayno, daytype):
    """Day-type filter from a per-detection day index (unix epoch day 0 =
    Thursday, so Monday=0 is (dayno+3)%7)."""
    if daytype == "all":
        return np.ones(dayno.size, dtype=bool)
    wd = (dayno.astype(np.int64) + 3) % 7
    return (wd < 5) if daytype == "weekday" else (wd >= 5)


def accumulate(paths, xb, yb, vs, kdir, nphase, fold, source, min_speed, period_s=0.0,
               daytype="all", dedup=False):
    """Single pass over the days. Returns two channels, keyed by linear cell:
       DIRECTION dhist[cell, phase, dir]  over MOVING detections only
       OCCUPANCY occ[cell, phase]         over ALL detections (present, incl. still)
    each with even/odd halves for the noise floor.

    Splitting direction from occupancy matters on a platform, where most
    detections are people standing (undefined heading): occupancy must count them,
    direction must not. The mover filter is `rho > min_speed` (min_speed=0 keeps
    every finite-heading detection, i.e. the ATC behaviour is unchanged).

    `fold` selects the periodic axis and its matching split (per detection, from
    each point's own day number, so a multi-day file splits correctly):
      * "hour"   -> phase = hour-of-day (nphase bins); split = calendar-day parity.
      * "dow"    -> phase = day-of-week (7 bins);      split = week parity.
      * "period" -> phase = position within a period_s cycle (e.g. the ~15/30-min
                    train cycle); split = alternate-cycle parity. This catches a
                    short, clock-locked oscillation that the hour fold averages
                    away."""
    x0, x1 = xb; y0, y1 = yb
    nx = int(math.ceil((x1 - x0) / vs))
    ny = int(math.ceil((y1 - y0) / vs))
    ncell = nx * ny
    dtheta = 2.0 * math.pi / kdir
    hour_span = 24.0 / nphase
    dshape = (ncell, nphase, kdir); oshape = (ncell, nphase)
    nd = ncell * nphase * kdir; no = ncell * nphase
    dhist = np.zeros(dshape); dhist_e = np.zeros(dshape); dhist_o = np.zeros(dshape)
    occ = np.zeros(oshape); occ_e = np.zeros(oshape); occ_o = np.zeros(oshape)
    # SPEED channel: same movers, speed binned into kspeed equal bins over
    # [min_speed, speed_cap); the same JSD machinery then asks whether the
    # SPEED DISTRIBUTION at a phase departs from the cell's all-phase speed
    # distribution -- the speed analogue of the direction-mix question.
    kspeed = 6
    speed_cap = 3.0
    sshape = (ncell, nphase, kspeed)
    ns = ncell * nphase * kspeed
    shist = np.zeros(sshape); shist_e = np.zeros(sshape); shist_o = np.zeros(sshape)
    for path in paths:
        s = _stream(source, path, xb, yb)
        if len(s.t) == 0:
            print(f"  [{os.path.basename(path)}] empty in bounds", flush=True)
            continue
        ix = np.clip(((s.x - x0) / vs).astype(np.int64), 0, nx - 1)
        iy = np.clip(((s.y - y0) / vs).astype(np.int64), 0, ny - 1)
        cell = ix * ny + iy
        dayno = np.floor(s.t / 86400.0)                    # per-detection day index
        if fold == "dow":
            iph = np.mod(dayno, 7).astype(np.int64)
            even = (np.mod(np.floor(dayno / 7.0), 2) == 0)  # week parity
        elif fold == "period":
            cyc = np.floor(s.t / period_s)                  # which cycle
            ph = np.mod(s.t, period_s)                      # seconds into cycle
            iph = np.clip((ph / (period_s / nphase)).astype(np.int64), 0, nphase - 1)
            even = (np.mod(cyc, 2) == 0)                     # alternate-cycle parity
        else:
            hod = np.mod(s.t / 3600.0, 24.0)
            iph = np.clip((hod / hour_span).astype(np.int64), 0, nphase - 1)
            even = (np.mod(dayno, 2) == 0)                  # calendar-day parity
        # Day-type filter: weekday and weekend timetables differ; mixing them
        # smears two schedules at the same time-of-day.
        keep = weekday_mask(dayno, daytype)
        # ---- occupancy: ALL detections (of the kept day-type) ----
        oflat = cell * nphase + iph
        occ += np.bincount(oflat[keep], minlength=no).reshape(oshape)
        occ_e += np.bincount(oflat[keep & even], minlength=no).reshape(oshape)
        occ_o += np.bincount(oflat[keep & ~even], minlength=no).reshape(oshape)
        # ---- direction: moving detections with a finite heading ----
        mv = np.isfinite(s.theta) & keep
        if min_speed > 0:
            mv &= np.isfinite(s.rho) & (s.rho > min_speed)
        ndedup = 0
        if mv.any():
            idir = np.mod(np.floor((s.theta[mv] + math.pi) / dtheta).astype(np.int64), kdir)
            dflat = (cell[mv] * nphase + iph[mv]) * kdir + idir
            evm = even[mv]
            if dedup:
                # One direction vote per (track, cell, phase, dir-bin) instead of
                # one per frame: a 10 Hz sensor re-detects the same crossing many
                # times, so per-frame counts inflate the effective sample size and
                # the split-half noise floor tracks bursts, not passages.
                # Occupancy is intentionally left per-frame (presence is
                # person-time; a loiterer IS present longer).
                _, tid = np.unique(s.id[mv], return_inverse=True)
                key = tid.astype(np.int64) * nd + dflat
                ukey, first = np.unique(key, return_index=True)
                dflat = ukey % nd
                evm = evm[first]
                ndedup = dflat.size
            dhist += np.bincount(dflat, minlength=nd).reshape(dshape)
            dhist_e += np.bincount(dflat[evm], minlength=nd).reshape(dshape)
            dhist_o += np.bincount(dflat[~evm], minlength=nd).reshape(dshape)
            # speed bins over the same movers (recompute the mover view: dedup
            # above may have reduced dflat/evm to unique votes, so rebuild from
            # the raw mask with its own dedup for the speed key)
            lo = max(min_speed, 0.0)
            sb = np.clip(((np.minimum(s.rho[mv], speed_cap - 1e-9) - lo)
                          / ((speed_cap - lo) / kspeed)).astype(np.int64), 0, kspeed - 1)
            sflat = (cell[mv] * nphase + iph[mv]) * kspeed + sb
            sevm = even[mv]
            if dedup:
                _, stid = np.unique(s.id[mv], return_inverse=True)
                skey = stid.astype(np.int64) * ns + sflat
                sukey, sfirst = np.unique(skey, return_index=True)
                sflat = sukey % ns
                sevm = sevm[sfirst]
            shist += np.bincount(sflat, minlength=ns).reshape(sshape)
            shist_e += np.bincount(sflat[sevm], minlength=ns).reshape(sshape)
            shist_o += np.bincount(sflat[~sevm], minlength=ns).reshape(sshape)
        print(f"  [{os.path.basename(path)}] {len(s.t):,} det "
              f"({int(mv.sum()):,} moving"
              + (f", {ndedup:,} deduped votes" if dedup else "") + ")", flush=True)
    return (dhist, dhist_e, dhist_o, occ, occ_e, occ_o,
            shist, shist_e, shist_o, (nx, ny))


def analyse(dhist, dhist_e, dhist_o, occ, occ_e, occ_o, topn):
    # ---- direction rhythm: ranked by MOVING traffic ----
    d_tot = dhist.sum(axis=(1, 2))
    dir_rows = []
    for c in [c for c in np.argsort(d_tot)[::-1][:topn] if d_tot[c] > 0]:
        h = dhist[c]                                       # (nphase, kdir)
        w = h.sum(axis=1); wsum = w.sum()
        if wsum <= 0:
            continue
        wn = w / wsum
        pbar = h.sum(axis=0)
        s_across = sum(wn[p] * js_divergence(h[p], pbar) for p in range(len(w)))
        he, ho = dhist_e[c], dhist_o[c]
        n_within = sum(wn[p] * js_divergence(he[p], ho[p]) for p in range(len(w)))
        dir_rows.append((c, d_tot[c], s_across, n_within,
                         s_across / n_within if n_within > 1e-9 else float("inf")))

    # ---- occupancy rhythm: ranked by ALL traffic ----
    o_tot = occ.sum(axis=1)
    occ_rows = []
    for c in [c for c in np.argsort(o_tot)[::-1][:topn] if o_tot[c] > 0]:
        w = occ[c]                                         # (nphase,)
        occ_across = js_divergence(w, np.ones_like(w))
        occ_within = js_divergence(occ_e[c], occ_o[c])
        occ_rows.append((c, o_tot[c], occ_across, occ_within,
                         occ_across / occ_within if occ_within > 1e-9 else float("inf")))
    return dir_rows, occ_rows


def report_bidirectionality(dhist, kdir, topn):
    """One-way vs two-way check: for the busiest moving cells, how much of the
    flow runs OPPOSITE the cell's dominant heading. ~0 means a one-way lane (a
    train cycle can only change how many pass, not the direction); a sizeable
    fraction means the spot is two-way, so a per-cycle reversal is possible."""
    d_tot = dhist.sum(axis=(1, 2))
    top = [c for c in np.argsort(d_tot)[::-1][:topn] if d_tot[c] > 0][:15]
    print("\n### ONE-WAY vs TWO-WAY (busiest moving spots)")
    print(f"  {'cell':>8} {'moving_det':>12} {'dom_deg':>8} {'reverse_frac':>12}  type")
    revs = []
    for c in top:
        m = dhist[c].sum(axis=0)                            # heading marginal
        dom = int(np.argmax(m))
        fwd = sum(m[(dom + j) % kdir] for j in (-1, 0, 1))  # within ~45 deg of dom
        opp = dom + kdir // 2
        bwd = sum(m[(opp + j) % kdir] for j in (-1, 0, 1))  # within ~45 deg of opposite
        rev = bwd / (fwd + bwd) if (fwd + bwd) > 0 else 0.0
        revs.append(rev)
        deg = (dom + 0.5) * 360.0 / kdir - 180.0
        print(f"  {c:>8} {int(m.sum()):>12,} {deg:>8.0f} {rev:>12.2f}  "
              f"{'two-way' if rev > 0.2 else 'one-way'}")
    if revs:
        print(f"  median reverse fraction (top {len(revs)}): {np.median(revs):.2f}  "
              f"(~0 one-way lanes; 0.3-0.5 two-way => reversal possible)")


def _report(name, rows, topn, phase="hour"):
    if not rows:
        print(f"\n### {name}: no cells with traffic"); return
    tot = np.array([r[1] for r in rows], float)
    across = np.array([r[2] for r in rows])
    within = np.array([r[3] for r in rows])
    w = tot / tot.sum()
    wa, wn = float((w * across).sum()), float((w * within).sum())
    print(f"\n### {name}  (top {len(rows)} cells, traffic-weighted)")
    print(f"  across-{phase} signal : {wa:.5f}")
    print(f"  within-{phase} noise  : {wn:.6f}")
    print(f"  rhythm index (S/N) : {wa / wn if wn > 1e-9 else float('inf'):.1f}")
    # Per-cell view: the weighted aggregate divides a mean signal by a mean
    # noise, so a rhythm confined to the busiest cells is diluted by the noisy
    # low-traffic tail. Report the distribution of per-cell ratios too.
    sn = np.array([r[4] for r in rows], float)
    fin = np.isfinite(sn)
    if fin.any():
        share3 = float(tot[sn > 3].sum() / tot.sum()) if (sn > 3).any() else 0.0
        print(f"  per-cell S/N: median {np.median(sn[fin]):.2f} | "
              f">1: {int((sn > 1).sum())}/{len(sn)} cells | "
              f">3: {int((sn > 3).sum())}/{len(sn)} cells "
              f"({100 * share3:.0f}% of traffic)")
        print("  (split-half null for this ratio is ~0.25-0.5, not 1: the noise "
              "halves carry twice the sampling variance of the full-data signal)")
    hot = sorted(rows, key=lambda r: r[1], reverse=True)[:min(10, len(rows))]
    print(f"  {'cell':>8} {'det':>12} {'across':>8} {'within':>8} {'S/N':>7}")
    for c, t, a, n, r in hot:
        print(f"  {c:>8} {int(t):>12,} {a:>8.4f} {n:>8.4f} {r:>7.2f}")


def resolve_source(args):
    """Fill `args.x_bounds` / `args.y_bounds` from the dataset's defaults where
    the caller left them unset, and return the day-file list.

    Shared with inspect_hour_profile.py: both scripts must resolve the same
    bounds, or the cell ids they report refer to different cells.
    """
    spec = DATASETS[args.source]
    bounds = spec["bounds"](args) if callable(spec["bounds"]) else spec["bounds"]
    xb, yb = bounds
    if args.x_bounds is None:
        args.x_bounds = list(xb)
    if args.y_bounds is None:
        args.y_bounds = list(yb)
    return args.days or spec["files"](args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, default="atc")
    ap.add_argument("--station", choices=["HB", "PB"], default="HB",
                    help="which HB station (HB=Havlickuv Brod, PB=Pardubice)")
    ap.add_argument("--npz-dir", default=os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "atc/npz"))
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets")), "eindhoven-station"),
                    help="parquet dir for eindhoven; tbd root for tbd")
    ap.add_argument("--days", nargs="+", default=None,
                    help="explicit day file paths (default: all ATC train days / "
                         "all Eindhoven parquet files)")
    ap.add_argument("--x-bounds", type=float, nargs=2, default=None)
    ap.add_argument("--y-bounds", type=float, nargs=2, default=None)
    ap.add_argument("--voxel-size-m", type=float, default=0.4)
    ap.add_argument("--dir-bins", type=int, default=8, help="heading bins (K)")
    ap.add_argument("--hour-bins", type=int, default=24, help="hour-of-day bins")
    ap.add_argument("--fold", choices=["hour", "dow", "period"], default="hour",
                    help="periodic axis: hour-of-day (daily), day-of-week (weekly), "
                         "or position within a short period_min cycle (e.g. trains)")
    ap.add_argument("--period-min", type=float, default=15.0,
                    help="cycle length in minutes for --fold period")
    ap.add_argument("--nphase", type=int, default=None,
                    help="phase bins for --fold period (default: 1 per minute)")
    ap.add_argument("--daytype", choices=["all", "weekday", "weekend"], default="all",
                    help="restrict to weekdays or weekends (timetables differ)")
    ap.add_argument("--min-speed-mps", type=float, default=0.0,
                    help="direction channel counts only detections faster than this "
                         "(exclude standing people; 0 = keep all, ATC behaviour)")
    ap.add_argument("--top-n", type=int, default=200, help="busiest cells to score")
    ap.add_argument("--dedup", action="store_true",
                    help="direction channel: one vote per (track, cell, phase) "
                         "instead of one per frame (burst/loiterer correction)")
    args = ap.parse_args()

    paths = resolve_source(args)

    period_s = 0.0
    if args.fold == "period":
        period_s = args.period_min * 60.0
        # >=6 bins so short (sub-6-min) periods are not degenerate.
        nphase = args.nphase or max(6, int(round(args.period_min)))
        phase = f"{args.period_min:g}min-cycle"
    elif args.fold == "dow":
        nphase = 7; phase = "weekday"
    else:
        nphase = args.hour_bins; phase = "hour"
    print(f"[periodicity] source={args.source} {len(paths)} files | fold={args.fold} "
          f"nphase={nphase} vox={args.voxel_size_m}m K={args.dir_bins} "
          f"min_speed={args.min_speed_mps} period_s={period_s:g} topN={args.top_n}")
    print(f"[periodicity] daytype={args.daytype} dedup={args.dedup}")
    (dh, dhe, dho, oc, oce, oco, sh, she, sho, (nx, ny)) = accumulate(
        paths, tuple(args.x_bounds), tuple(args.y_bounds), args.voxel_size_m,
        args.dir_bins, nphase, args.fold, args.source, args.min_speed_mps, period_s,
        args.daytype, args.dedup)
    print(f"[periodicity] grid {nx}x{ny}, moving det={int(dh.sum()):,}, "
          f"all det={int(oc.sum()):,}")
    dir_rows, occ_rows = analyse(dh, dhe, dho, oc, oce, oco, args.top_n)
    spd_rows, _ = analyse(sh, she, sho, oc, oce, oco, args.top_n)
    _report("DIRECTION MIX (fix 3.1)", dir_rows, args.top_n, phase)
    _report("SPEED MIX (binned speed distribution)", spd_rows, args.top_n, phase)
    _report("OCCUPANCY (positive control)", occ_rows, args.top_n, phase)
    report_bidirectionality(dh, args.dir_bins, args.top_n)
    print("\nRead: S/N >> 1 => rhythm beyond sampling noise. Occupancy is the "
          "known-positive control; direction is the open question.")


if __name__ == "__main__":
    main()
