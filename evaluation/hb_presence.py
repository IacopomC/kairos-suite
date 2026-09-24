#!/usr/bin/env python3
r"""Presence-estimation read on the Havlickuv Brod (HB) station concourses.

Same protocol as the ATC presence evaluation: estimators are fit on the
training span and frozen; each held-out day is scanned on a regular time grid;
at every grid time t each training-covered cell's predicted probability of the
windowed event "occupied within [t, t+dt)" is scored against the realized
occupancy of that window, per horizon dt. Scores are Bernoulli log-loss with
the reliability/resolution decomposition (PresenceAccumulator, unchanged).

Estimators (all fit on the identical windowed-event stream):
  base_rate  global per-horizon occupancy frequency of the training span
             (calibrated by construction, no cell information)
  fremen     per-cell spectral occupancy (FreMEn): DC plus one complex
             coefficient per candidate period, the same primitive as
             FremenOccupancy in online_metrics.py

The event grid: windows start at multiples of --grid-step and must lie fully
inside their calendar day (windows crossing midnight would need the next
day's file); the horizon, not the grid step, defines the event. The fit
accumulates each cell's spectral coefficients over OCCUPIED grid steps only,
as geometric sums over merged occupied runs, so an 11-month span at a 5 s
step is a single cheap pass. A --selftest mode checks the run construction
against a brute-force window scan and the fit/prediction against
FremenOccupancy on the same event stream.

Only dates for which every camera of the station has a file are used, so the
spatial coverage of the stream is constant across the span.

Usage:
    python3 hb_presence.py --station HB \
        --train-start 2022-09-05 --train-end 2023-04-23 \
        --test-start 2023-04-24 --test-end 2023-05-21 \
        --horizons 5 10 60 300 600 --grid-step 5

    python3 hb_presence.py --station HB --selftest    # check the run
                                                      # construction and the fit

Every estimator is fit on the identical windowed-event stream and frozen before
the held-out days are scanned, so the comparison is of estimators, not of
protocols. Keep --grid-step well below the shortest horizon: the horizon, not
the grid step, defines the event.
"""
import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "grounded_sim"))

from online_metrics import PresenceAccumulator, FremenOccupancy, htag, _PRESENCE_EPS  # noqa: E402

STATION_BOUNDS = {"HB": ([-2.0, 35.0], [-32.0, 2.0]),
                  "PB": ([-4.0, 82.0], [-22.0, 4.0])}
DAY_S = 86400


# ---------------------------------------------------------------- data layout

def station_dates(root, station):
    """Dates covered by EVERY camera of the station -> {date: [files]}."""
    cams = sorted(glob.glob(os.path.join(root, f"{station}-CAM*")))
    per_cam = []
    for cam in cams:
        d = {}
        for f in glob.glob(os.path.join(cam, "*_trans.csv")):
            iso = os.path.basename(f).split("_")[1]
            d[date.fromisoformat(iso)] = f
        per_cam.append(d)
    common = set(per_cam[0])
    for d in per_cam[1:]:
        common &= set(d)
    return {dt: [d[dt] for d in per_cam] for dt in sorted(common)}, len(cams)


def load_day(files, xb, yb):
    """Concatenated (t, x, y) for one date (all cameras, bounds applied)."""
    from hb_stream import HBStream
    ts, xs, ys = [], [], []
    for f in files:
        s = HBStream(f, x_bounds=xb, y_bounds=yb)
        ts.append(s.t); xs.append(s.x); ys.append(s.y)
    return np.concatenate(ts), np.concatenate(xs), np.concatenate(ys)


# ------------------------------------------------------- occupied-window runs

def n_steps_day(h, g):
    """Windows of length h fully inside one day, at start-step g."""
    return (DAY_S - int(math.ceil(h))) // g + 1


def occupied_runs(cell, t, day0, h, g):
    """Merged occupied-step runs per cell for one day and horizon.

    A window starting at step k (absolute time k*g) is occupied by a detection
    at time tau iff k in [floor((tau-h)/g)+1, floor(tau/g)], clamped to the
    day's fully-contained step range. Input must be sorted by (cell, t).
    Returns (run_cell, run_a, run_b) in ABSOLUTE step indices.
    """
    k0 = day0 // g
    k1 = k0 + n_steps_day(h, g) - 1
    a = np.floor_divide(t - h, g).astype(np.int64) + 1
    b = np.floor_divide(t, g).astype(np.int64)
    np.clip(a, k0, k1 + 1, out=a)          # a may exceed b -> empty, dropped
    np.clip(b, k0 - 1, k1, out=b)
    ok = a <= b
    cell, a, b = cell[ok], a[ok], b[ok]
    if cell.size == 0:
        return cell, a, b
    # t ascending within cell => a and b ascending within cell: chain-merge.
    new = np.ones(cell.size, dtype=bool)
    new[1:] = (cell[1:] != cell[:-1]) | (a[1:] > b[:-1] + 1)
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:] - 1, cell.size - 1)
    return cell[starts], a[starts], b[ends]


def geometric_sum(a, b, z_arg):
    """sum_{k=a}^{b} exp(-1j * z_arg * k), vectorized over runs, per frequency.

    a, b: (R,) int; z_arg: scalar omega*g. Returns (R,) complex.
    """
    n = (b - a + 1).astype(np.float64)
    za = np.exp(-1j * (z_arg * a.astype(np.float64)))
    z = np.exp(-1j * z_arg)
    if abs(1.0 - z) < 1e-12:
        return za * n
    return za * (1.0 - np.exp(-1j * (z_arg * n))) / (1.0 - z)


# ----------------------------------------------------------------- the model

class SpectralOccupancy:
    """Per-cell FreMEn occupancy fit from occupied runs (fixed cell universe).

    Same model and prediction formula as online_metrics.FremenOccupancy
    (gamma0 + sum_f 2|c_f| cos(omega_f t + angle c_f), clipped), fit as
    geometric sums over occupied runs instead of a per-step stream.
    """

    def __init__(self, n_cells, periods_s, horizons, g):
        self.omegas = np.array([2.0 * math.pi / float(P) for P in periods_s])
        self.horizons = tuple(float(h) for h in horizons)
        self.g = g
        self.n_steps = {h: 0 for h in self.horizons}            # total train steps
        self.n_occ = {h: np.zeros(n_cells) for h in self.horizons}
        self.S = {h: np.zeros((n_cells, self.omegas.size), dtype=np.complex128)
                  for h in self.horizons}

    def enable_selection(self, stride=12):
        """Prequential order selection (the FreMEn model-order choice, done
        leak-free): before each training day is folded in,
        score_day_prequential evaluates the running fit's nested orders
        (components ranked by current amplitude, largest first) on that day's
        windows, subsampled every `stride` grid steps; the accumulated
        log-loss picks each cell's order per horizon. Order l uses the l most
        prominent components; order 0 is the constant rate."""
        self._sel_stride = int(stride)
        F = self.omegas.size
        n_cells = self.n_occ[self.horizons[0]].size
        self._sel_ll = {h: np.zeros((n_cells, F + 1)) for h in self.horizons}
        self.sel_order = None

    def score_day_prequential(self, runs_by_h, day0):
        if not hasattr(self, "_sel_ll"):
            return
        eps = _PRESENCE_EPS
        F = self.omegas.size
        for h in self.horizons:
            n = float(self.n_steps[h])
            if n <= 0:
                continue                    # no fit yet: day 1 is not scored
            K = n_steps_day(h, self.g)
            n_cells = self.n_occ[h].size
            cell, a, b = runs_by_h[h]
            k0 = day0 // self.g
            # occupied mask on the subgrid (runs are merged and disjoint, so
            # the start/end delta trick stays within int8)
            delta = np.zeros((n_cells, K + 1), dtype=np.int8)
            if cell.size:
                np.add.at(delta, (cell, a - k0), 1)
                np.add.at(delta, (cell, b - k0 + 1), -1)
            y = (np.cumsum(delta[:, :K], axis=1) > 0)[:, ::self._sel_stride]
            ks = np.arange(K)[::self._sel_stride]
            t = (k0 + ks) * float(self.g)
            g0 = self.n_occ[h] / n
            c = self.S[h] / n
            mag, phase = np.abs(c), np.angle(c)
            rank = np.argsort(-mag, axis=1)
            rows = np.arange(n_cells)
            p = np.repeat(g0[:, None], t.size, axis=1)
            for l in range(F + 1):
                pc = np.clip(p, eps, 1 - eps)
                ll = (y * np.log(pc) + (~y) * np.log(1 - pc)).sum(axis=1)
                self._sel_ll[h][:, l] += ll
                if l < F:
                    fi = rank[:, l]
                    p = p + 2.0 * mag[rows, fi][:, None] * np.cos(
                        self.omegas[fi][:, None] * t[None, :]
                        + phase[rows, fi][:, None])

    def finalize_selection(self):
        """Pick each cell's prequentially best order and mask the final fit
        to its top components; call after finalize(). predict/predict_at then
        use the selection (order 0 reduces to the constant rate)."""
        if not hasattr(self, "_sel_ll"):
            return
        self.sel_order, self.mag_full = {}, {}
        for h in self.horizons:
            sel = np.argmax(self._sel_ll[h], axis=1)
            self.sel_order[h] = sel
            rank = np.argsort(-self.mag[h], axis=1)
            keep = np.zeros(self.mag[h].shape, dtype=bool)
            rows = np.arange(sel.size)
            for l in range(1, self.omegas.size + 1):
                m = sel >= l
                keep[rows[m], rank[m, l - 1]] = True
            self.mag_full[h] = self.mag[h].copy()
            self.mag[h] = np.where(keep, self.mag[h], 0.0)

    def add_day(self, runs_by_h, day0):
        for h in self.horizons:
            self.n_steps[h] += n_steps_day(h, self.g)
            cell, a, b = runs_by_h[h]
            if cell.size == 0:
                continue
            np.add.at(self.n_occ[h], cell, (b - a + 1).astype(np.float64))
            for fi, om in enumerate(self.omegas):
                np.add.at(self.S[h], (cell, fi), geometric_sum(a, b, om * self.g))

    def finalize(self):
        self.g0, self.mag, self.phase, self.global_rate = {}, {}, {}, {}
        for h in self.horizons:
            n = float(self.n_steps[h])
            self.g0[h] = self.n_occ[h] / n
            c = self.S[h] / n
            self.mag[h], self.phase[h] = np.abs(c), np.angle(c)
            self.global_rate[h] = float(self.n_occ[h].sum() / (n * self.n_occ[h].size))

    def predict(self, rows, t_steps, h):
        """(len(rows), len(t_steps)) probabilities at absolute step indices."""
        t = t_steps.astype(np.float64) * self.g
        p = np.repeat(self.g0[h][rows][:, None], t.size, axis=1)
        for fi, om in enumerate(self.omegas):
            p += 2.0 * self.mag[h][rows, fi][:, None] \
                * np.cos(om * t[None, :] + self.phase[h][rows, fi][:, None])
        return np.clip(p, _PRESENCE_EPS, 1.0 - _PRESENCE_EPS)

    def predict_at(self, rows, t, h):
        """Probabilities for `rows` at one absolute time t (seconds); the
        harmonic model is continuous in t, so t need not sit on the fit grid."""
        p = self.g0[h][rows].copy()
        for fi, om in enumerate(self.omegas):
            p += 2.0 * self.mag[h][rows, fi] \
                * np.cos(om * float(t) + self.phase[h][rows, fi])
        return np.clip(p, _PRESENCE_EPS, 1.0 - _PRESENCE_EPS)


# ------------------------------------------------------------------ selftest

def selftest(g=5):
    rng = np.random.default_rng(0)
    horizons = (5.0, 10.0, 60.0)
    day0 = 730 * DAY_S                      # arbitrary absolute day
    n_cells = 3
    n_det = 400
    t = np.sort(day0 + rng.integers(0, DAY_S - 100, n_det).astype(np.float64))
    cell = rng.integers(0, n_cells, n_det).astype(np.int64)
    order = np.lexsort((t, cell))
    cell, t = cell[order], t[order]

    # (a) occupied runs against a brute-force window scan
    for h in horizons:
        rc, ra, rb = occupied_runs(cell, t, day0, h, g)
        ns = n_steps_day(h, g)
        y = np.zeros((n_cells, ns))
        for c, a, b in zip(rc, ra - day0 // g, rb - day0 // g):
            y[c, a:b + 1] = 1.0
        k = day0 // g + np.arange(ns)
        for c in range(n_cells):
            tc = t[cell == c]
            brute = np.array([np.any((tc >= kk * g) & (tc < kk * g + h))
                              for kk in k], dtype=float)
            assert np.array_equal(y[c], brute), f"run construction differs (h={h})"

    # (b) fit + prediction parity against FremenOccupancy on the same events
    periods = (3600.0, 7200.0)
    model = SpectralOccupancy(n_cells, periods, horizons, g)
    runs = {h: occupied_runs(cell, t, day0, h, g) for h in horizons}
    model.add_day(runs, day0)
    model.finalize()
    ref = FremenOccupancy(periods, horizons=horizons)
    ns_max = n_steps_day(min(horizons), g)
    for kk in range(ns_max):
        tt = float((day0 // g + kk) * g)
        y_by_h = {}
        for h in horizons:
            if kk >= n_steps_day(h, g):
                y_by_h[h] = np.zeros(n_cells)   # ref needs equal-length streams
                continue
            rc, ra, rb = runs[h]
            step = day0 // g + kk
            y_by_h[h] = np.array([np.any((rc == c) & (ra <= step) & (step <= rb))
                                  for c in range(n_cells)], dtype=float)
        ref.fit_update(np.arange(n_cells), np.zeros(n_cells, dtype=np.int64),
                       np.zeros(n_cells, dtype=np.int64), tt, y_by_h)
    ref.finalize()
    h = 5.0     # the horizon whose ref stream had no padding rows
    md = ref._model[h]
    assert np.allclose(md["g0"], model.g0[h], atol=1e-12)
    assert np.allclose(md["mag"], model.mag[h], atol=1e-9)
    dphi = (md["phase"] - model.phase[h] + math.pi) % (2 * math.pi) - math.pi
    assert np.allclose(np.where(md["mag"] > 1e-12, dphi, 0.0), 0.0, atol=1e-6)
    t_q = day0 // g + np.arange(0, 200, 7)
    p_mine = model.predict(np.arange(n_cells), t_q, h)
    p_ref = np.stack([ref.predict(np.arange(n_cells),
                                  np.zeros(n_cells, dtype=np.int64),
                                  np.zeros(n_cells, dtype=np.int64),
                                  float(kk * g), h) for kk in t_q], axis=1)
    assert np.allclose(p_mine, p_ref, atol=1e-9)
    print("[selftest] runs vs brute force + fit/predict parity vs FremenOccupancy: OK")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root",
                    default=os.path.join(
                        os.environ.get("DATA_ROOT",
                                       os.path.expanduser("~/datasets")),
                        "hb-station", "Data-2"))
    ap.add_argument("--station", choices=["HB", "PB"], default="HB")
    ap.add_argument("--voxel", type=float, default=0.4)
    ap.add_argument("--grid-step", type=int, default=5,
                    help="window start grid (s); the horizon defines the event")
    ap.add_argument("--horizons", type=float, nargs="+",
                    default=[5.0, 10.0, 60.0, 300.0, 600.0])
    ap.add_argument("--periods-h", type=float, nargs="+", default=[1.0, 12.0, 24.0, 168.0])
    ap.add_argument("--train-start", default="2022-09-05")
    ap.add_argument("--train-end", default="2023-04-23")
    ap.add_argument("--test-start", default="2023-04-24")
    ap.add_argument("--test-end", default="2023-05-21")
    ap.add_argument("--chunk-cells", type=int, default=128)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max-train-days", type=int, default=0, help="smoke cap")
    ap.add_argument("--max-test-days", type=int, default=0, help="smoke cap")
    args = ap.parse_args()

    if args.selftest:
        selftest(g=args.grid_step)
        return

    assert DAY_S % args.grid_step == 0
    xb, yb = STATION_BOUNDS[args.station]
    dates, n_cams = station_dates(args.root, args.station)
    span = {k: v for k, v in dates.items()}
    tr0, tr1 = date.fromisoformat(args.train_start), date.fromisoformat(args.train_end)
    te0, te1 = date.fromisoformat(args.test_start), date.fromisoformat(args.test_end)
    train_days = [d for d in span if tr0 <= d <= tr1]
    test_days = [d for d in span if te0 <= d <= te1]
    if args.max_train_days:
        train_days = train_days[:args.max_train_days]
    if args.max_test_days:
        test_days = test_days[:args.max_test_days]
    g, H = args.grid_step, tuple(args.horizons)
    periods_s = [h * 3600.0 for h in args.periods_h]

    print(f"[hb-presence] station={args.station} cams={n_cams} voxel={args.voxel} "
          f"grid={g}s horizons={list(H)} periods_h={args.periods_h}")
    print(f"[hb-presence] train {train_days[0]}..{train_days[-1]} ({len(train_days)} days) "
          f"| test {test_days[0]}..{test_days[-1]} ({len(test_days)} days) "
          f"(dates with all {n_cams} cameras only)")
    sys.stdout.flush()

    def cell_of(x, y):
        return (np.floor((x - xb[0]) / args.voxel).astype(np.int64),
                np.floor((y - yb[0]) / args.voxel).astype(np.int64))

    nx = int(math.ceil((xb[1] - xb[0]) / args.voxel))

    # ---- pass 1: discover the cell universe + count detections
    key_row = {}
    for d in train_days:
        t, x, y = load_day(span[d], xb, yb)
        cx, cy = cell_of(x, y)
        for k in np.unique(cx * nx + cy):
            key_row.setdefault(int(k), len(key_row))
        print(f"  [scan] {d}: {t.size} detections, {len(key_row)} cells so far")
        sys.stdout.flush()
    n_cells = len(key_row)
    print(f"[hb-presence] cell universe: {n_cells} train-covered cells")
    sys.stdout.flush()

    # ---- pass 2: fit (occupied runs -> geometric spectral sums)
    model = SpectralOccupancy(n_cells, periods_s, H, g)
    lut = np.full(max(key_row) + 1, -1, dtype=np.int64)
    for k, r in key_row.items():
        lut[k] = r

    def day_rows(d):
        t, x, y = load_day(span[d], xb, yb)
        cx, cy = cell_of(x, y)
        rows = lut[np.clip(cx * nx + cy, 0, lut.size - 1)]
        rows = np.where((cx * nx + cy > lut.size - 1) | (rows < 0), -1, rows)
        return t, rows

    for d in train_days:
        t, rows = day_rows(d)
        m = rows >= 0
        t, rows = t[m], rows[m]
        order = np.lexsort((t, rows))
        t, rows = t[order], rows[order]
        day0 = int((d - date(1970, 1, 1)).days) * DAY_S
        model.add_day({h: occupied_runs(rows, t, day0, h, g) for h in H}, day0)
    model.finalize()
    rates = {h: model.global_rate[h] for h in H}
    print("[hb-presence] TRAIN occupancy rate " +
          " ".join(f"{htag(h)}s={rates[h]:.6f}" for h in H))
    sys.stdout.flush()

    # ---- pass 3: frozen scan of the held-out days
    acc = PresenceAccumulator(estimators=("fremen",), horizons=H,
                              base_rates=rates)
    n_uncov = 0
    for d in test_days:
        t, rows = day_rows(d)
        n_uncov += int((rows < 0).sum())
        m = rows >= 0
        t, rows = t[m], rows[m]
        order = np.lexsort((t, rows))
        t, rows = t[order], rows[order]
        day0 = int((d - date(1970, 1, 1)).days) * DAY_S
        runs = {h: occupied_runs(rows, t, day0, h, g) for h in H}
        for c0 in range(0, n_cells, args.chunk_cells):
            rows_chunk = np.arange(c0, min(c0 + args.chunk_cells, n_cells))
            y_by_tag, preds = {}, {"fremen": {}}
            for h in H:
                ns = n_steps_day(h, g)
                k0 = day0 // g
                rc, ra, rb = runs[h]
                sel = (rc >= c0) & (rc < c0 + rows_chunk.size)
                diff = np.zeros((rows_chunk.size, ns + 1))
                np.add.at(diff, (rc[sel] - c0, ra[sel] - k0), 1.0)
                np.add.at(diff, (rc[sel] - c0, rb[sel] - k0 + 1), -1.0)
                y = (np.cumsum(diff[:, :-1], axis=1) > 0).astype(float)
                tag = htag(h)
                y_by_tag[tag] = y.ravel()
                t_steps = k0 + np.arange(ns)
                preds["fremen"][tag] = model.predict(rows_chunk, t_steps, h).ravel()
            acc.update(y_by_tag, preds)
        print(f"  [test] {d}: scanned ({t.size} detections)")
        sys.stdout.flush()
    if n_uncov:
        print(f"[hb-presence] test detections outside the train-covered "
              f"cell universe (not scanned): {n_uncov}")

    # ---- report
    res = acc.result()
    print("----- PRESENCE (Bernoulli-LL / reliability / resolution) -----")
    for h in H:
        r = res[htag(h)]
        print(f"  --- horizon {htag(h)}s ---  occupancy_rate={r['occupancy_rate']:.4f}"
              f"  uncertainty(floor)={r['uncertainty']:.5f}  (n={r['n']})")
        print(f"  {'estimator':>17} {'BernLL':>9} {'reliab':>9} {'resol':>9}")
        for nm in ("base_rate", "fremen"):
            e = r[nm]
            print(f"  {nm:>17} {e['bern_ll']:9.4f} {e['reliability']:9.5f} "
                  f"{e['resolution']:9.5f}")
    print("[hb-presence] done")


if __name__ == "__main__":
    main()
