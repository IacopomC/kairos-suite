#!/usr/bin/env python3
r"""Online evaluator: score a trained model over the held-out test set and
accumulate every metric in ONE streaming pass -- no per-detection files written.

One standard across datasets. Two seams:
  * method scorer  -- score a chunk of detections -> per-detection arrays
                      (via the pipeline's score_observations).
  * dataset source -- stream the held-out detections in chunks
                      (atc: the held-out day npz; tbd: its tracks).
The metric math lives in online_metrics.py (FlowAccumulator, CalibrationAccumulator,
CoverageStratifiedAccumulator, CouplingAccumulator, PresenceAccumulator).

Flow / calibration / coverage-stratified / coupling run in one detection-streaming
pass; presence runs a separate robot-coverage replay (--presence).

Examples:
  # ATC / KAIROS flow+calibration+coverage+coupling, full held-out set:
  python3 evaluate.py --method kairos --dataset atc --config-name atc \
      --graph /data/atc/graph/atc_graph.json --scene-dir ../grounded_sim/stub_scene \
      --flow-state results/atc/train/flow_state.bin \
      --days /data/atc/npz/atc-20130901.npz ...

  # ATC presence (occupancy) via robot-coverage replay:
  python3 evaluate.py ... --presence --days /data/atc/npz/atc-20130901.npz ...
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))       # so `baselines` imports as a package
import baselines  # noqa: E402
from online_metrics import (FlowAccumulator, CalibrationAccumulator,  # noqa: E402
                            CoverageStratifiedAccumulator, CouplingAccumulator,
                            PresenceAccumulator, TemporalStratifiedAccumulator,
                            LcMlpdDeltaAccumulator, PhaseStratifiedAccumulator,
                            HeadingDistributionAccumulator,
                            HistogramHeadingAccumulator,
                            BinnedControlAccumulator, PassageWeightedAccumulator,
                            CellMeanAccumulator, WeightCalibrationAccumulator,
                            FremenOccupancy, DEFAULT_SPEED_CAP_MPS, htag,
                            _PRESENCE_EPS)


# ---------------------------------------------------------------- method scorers
# ---------------------------------------------------------------- dataset sources
def atc_source(days, bounds, chunk, minutes=0.0):
    """Yield (t_s, xyz, theta, rho) chunks of the held-out ATC days (god's-eye,
    all detections), streamed from the npz so nothing is buffered whole. A
    positive `minutes` caps each day to its first `minutes` minutes, matching
    the cap the presence replay applies (a capped smoke run must not
    silently score full days)."""
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from atc_stream import AtcStream
    xb = (bounds[0], bounds[1]); yb = (bounds[2], bounds[3])
    for day in days:
        s = AtcStream(day, x_bounds=xb, y_bounds=yb)
        n = len(s.t)
        if minutes > 0.0 and n:
            n = int(np.searchsorted(s.t, s.t_min + minutes * 60.0, side="right"))
            print(f"  [source] {pathlib.Path(day).name}: capped to first "
                  f"{minutes:g} min ({n:,} detections)", flush=True)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            t_s = s.t[lo:hi].tolist()
            xyz = [[float(s.x[i]), float(s.y[i]), 0.0] for i in range(lo, hi)]
            yield (t_s, xyz, s.theta[lo:hi].tolist(), s.rho[lo:hi].tolist(),
                   s.id[lo:hi].astype(np.int64).tolist())


def hb_source(root, station, dates, bounds, chunk, interp_hz, gap_max_s,
                 minutes=0.0):
    """Yield (t_s, xyz, theta, rho, ids) chunks of held-out station dates
    (god's-eye, all detections of the 10 Hz interpolated stream), the hb
    counterpart of atc_source. Dates must be chronological."""
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from hb_stream import HBDayStream, hb_day_files
    xb = (bounds[0], bounds[1]); yb = (bounds[2], bounds[3])
    for day in dates:
        files = hb_day_files(root, station, day)
        if not files:
            raise SystemExit(f"[evaluate] no hb files for {station} {day}")
        s = HBDayStream(files, x_bounds=xb, y_bounds=yb,
                           interp_hz=interp_hz, gap_max_s=gap_max_s)
        n = len(s.t)
        if minutes > 0.0 and n:
            n = int(np.searchsorted(s.t, s.t_min + minutes * 60.0, side="right"))
            print(f"  [source] {day}: capped to first {minutes:g} min "
                  f"({n:,} detections)", flush=True)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            xyz = [[float(s.x[i]), float(s.y[i]), 0.0] for i in range(lo, hi)]
            yield (s.t[lo:hi].tolist(), xyz, s.theta[lo:hi].tolist(),
                   s.rho[lo:hi].tolist(), s.id[lo:hi].astype(np.int64).tolist())


def tbd_source(track_csvs, chunk, minutes=0.0):
    """Yield (t_s, xyz, theta, rho) chunks of ALL held-out TBD detections
    (god's-eye: every annotated detection across the held-out scenes, scored
    against the frozen trained map), streamed from each scene's tracks.csv.
    Columns: t_ns, ped_id, x_world, y_world, z_world, theta_rad, rho_mps."""
    import csv
    t_s, xyz, theta, rho, ids = [], [], [], [], []
    for path in track_csvs:
        t_cap = None
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                t_r = int(r["t_ns"]) / 1e9
                if minutes > 0.0:
                    if t_cap is None:
                        t_cap = t_r + minutes * 60.0  # per-scene cap from first row
                    if t_r > t_cap:
                        break
                t_s.append(t_r)
                xyz.append([float(r["x_world"]), float(r["y_world"]),
                            float(r.get("z_world", 0.0) or 0.0)])
                theta.append(float(r["theta_rad"])); rho.append(float(r["rho_mps"]))
                ids.append(int(r.get("ped_id", -1) or -1))
                if len(t_s) >= chunk:
                    yield t_s, xyz, theta, rho, ids
                    t_s, xyz, theta, rho, ids = [], [], [], [], []
    if t_s:
        yield t_s, xyz, theta, rho, ids


def _graph_bounds(graph):
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from robot_sim import load_nav_graph
    pos, _, _, _ = load_nav_graph(graph)
    return (pos[:, 0].min() - 6, pos[:, 0].max() + 6,
            pos[:, 1].min() - 6, pos[:, 1].max() + 6)


# ---------------------------------------------------------------- presence replay
def hb_stream_opener(root, station, interp_hz, gap_max_s):
    """Day opener for the HB station read: `day` is a YYYY-MM-DD date, the
    stream is the station's concatenated cameras interpolated per track to
    `interp_hz` (HBDayStream), which restores the ATC sampling condition
    (a voxel crossing leaves detections) for training and ground truth alike."""
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from hb_stream import HBDayStream, hb_day_files

    def open_day(day, xb, yb):
        files = hb_day_files(root, station, day)
        if not files:
            raise SystemExit(f"[evaluate] no hb files for {station} {day}")
        return HBDayStream(files, x_bounds=xb, y_bounds=yb,
                              interp_hz=interp_hz, gap_max_s=gap_max_s)
    return open_day


class _HBFremenAdapter:
    """FremenOccupancy-compatible .predict over a hb_presence
    SpectralOccupancy fit (cells unseen in training get the global rate)."""

    def __init__(self, model, row_of):
        self.model = model; self.row_of = row_of

    def predict(self, vx, vy, vz, t, h):
        rows = np.array([self.row_of.get((int(a), int(b)), -1)
                         for a, b in zip(vx, vy)], dtype=np.int64)
        p = np.full(rows.size, self.model.global_rate[float(h)])
        seen = rows >= 0
        if seen.any():
            p[seen] = self.model.predict_at(rows[seen], float(t), float(h))
        return np.clip(p, _PRESENCE_EPS, 1.0 - _PRESENCE_EPS)


def fit_fremen_hb(graph, train_days, stream_open, bounds, periods_s, grid_dt,
                     horizons=(5.0, 10.0), hours=None, cache=None, select=False):
    """Full-visibility FreMEn fit for the station read. The station has no
    robot, so there is no coverage replay: the fit consumes the identical
    windowed events over the authored graph's cell universe, accumulated as
    geometric sums over merged occupied runs (hb_presence.SpectralOccupancy,
    whose parity against FremenOccupancy is unit-tested there). day0 is each
    date's UTC midnight; windows lie fully inside their calendar day."""
    from datetime import date as _date
    from hb_presence import SpectralOccupancy, occupied_runs
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from robot_sim import load_nav_graph
    VS = 0.4
    g = int(round(grid_dt))
    assert abs(grid_dt - g) < 1e-9 and 86400 % g == 0, \
        "hb fremen fit needs an integer grid_dt dividing the day"
    pos, _, _, _ = load_nav_graph(graph)
    kvx = np.floor(pos[:, 0] / VS).astype(np.int64)
    kvy = np.floor(pos[:, 1] / VS).astype(np.int64)
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(kvx, kvy))}
    horizons = [float(h) for h in horizons]
    if cache is not None and hours is None and pathlib.Path(cache).exists():
        z = np.load(cache)
        assert list(z["periods_s"]) == [float(P) for P in periods_s] \
            and int(z["grid_dt"]) == g and list(z["horizons"]) == horizons \
            and int(z["n_cells"]) == len(row_of) \
            and int(z.get("selected", 0)) == int(select), "fremen cache mismatch"
        model = SpectralOccupancy(len(row_of), periods_s, horizons, g)
        model.g0, model.mag, model.phase, model.global_rate = {}, {}, {}, {}
        for h in horizons:
            t = htag(h)
            model.g0[h] = z[f"g0_{t}"]
            model.mag[h] = z[f"mag_{t}"]
            model.phase[h] = z[f"phase_{t}"]
            model.global_rate[h] = float(z[f"rate_{t}"])
        print(f"  [fremen-fit] loaded cached fit from {cache}")
        train_rate = {h: model.global_rate[h] for h in horizons}
        return _HBFremenAdapter(model, row_of), train_rate
    model = SpectralOccupancy(len(row_of), periods_s, horizons, g)
    if select:
        model.enable_selection()
    xb = (bounds[0], bounds[1]); yb = (bounds[2], bounds[3])
    n_out = 0
    # Hour-restricted train occupancy rate (the fair constant floor when the
    # scoring pass is restricted to the same hours). The spectral fit itself
    # always consumes the full day.
    hr_occ = {h: 0.0 for h in horizons}
    hr_tot = {h: 0.0 for h in horizons}
    for day in train_days:
        stream = stream_open(day, xb, yb)
        vx = np.floor(stream.x / VS).astype(np.int64)
        vy = np.floor(stream.y / VS).astype(np.int64)
        rows = np.array([row_of.get((int(a), int(b)), -1)
                         for a, b in zip(vx, vy)], dtype=np.int64)
        m = rows >= 0
        n_out += int((~m).sum())
        t, rows = stream.t[m], rows[m]
        o = np.lexsort((t, rows))
        t, rows = t[o], rows[o]
        day0 = int((_date.fromisoformat(day) - _date(1970, 1, 1)).days) * 86400
        runs_by_h = {h: occupied_runs(rows, t, day0, h, g) for h in horizons}
        if select:
            model.score_day_prequential(runs_by_h, day0)
        model.add_day(runs_by_h, day0)
        if hours is not None:
            lo = day0 // g + int(hours[0] * 3600) // g
            hi = day0 // g + int(hours[1] * 3600) // g
            for h in horizons:
                cell, a, b = runs_by_h[h]
                if cell.size:
                    hr_occ[h] += float(np.maximum(
                        0, np.minimum(b, hi - 1) - np.maximum(a, lo) + 1).sum())
                hr_tot[h] += float(len(row_of) * (hi - lo))
        print(f"  [fremen-fit] {day}: {t.size} detections", flush=True)
    model.finalize()
    if select:
        model.finalize_selection()
        for h in horizons:
            hist = np.bincount(model.sel_order[h], minlength=len(periods_s) + 1)
            print(f"  [fremen-fit] prequential order selection @ {htag(h)}s: "
                  + " ".join(f"l{l}={int(c)}" for l, c in enumerate(hist)))
    if n_out:
        print(f"  [fremen-fit] detections outside the graph universe: {n_out}")
    # Which candidate period carries the fit: mean spectral magnitude per
    # period (compare against the kairos state's validated gate orders).
    for h in horizons:
        line = " ".join(f"{float(P)/3600:g}h={model.mag[h][:, fi].mean():.5f}"
                        for fi, P in enumerate(periods_s))
        print(f"  [fremen-fit] mean |c| per period @ {htag(h)}s: {line} "
              f"(g0 mean {model.g0[h].mean():.5f})")
    if hours is not None:
        train_rate = {h: hr_occ[h] / hr_tot[h] for h in horizons}
        print(f"  [fremen-fit] TRAIN occupancy rate restricted to "
              f"{hours[0]:g}-{hours[1]:g} h " +
              " ".join(f"{htag(h)}s={train_rate[h]}" for h in horizons))
    else:
        train_rate = {h: model.global_rate[h] for h in horizons}
    print("  [fremen-fit] TRAIN occupancy rate (full day) " +
          " ".join(f"{htag(h)}s={model.global_rate[h]}" for h in horizons))
    if cache is not None and hours is None:
        payload = {"periods_s": np.asarray([float(P) for P in periods_s]),
                   "grid_dt": g, "horizons": np.asarray(horizons),
                   "n_cells": len(row_of), "selected": int(select)}
        for h in horizons:
            t = htag(h)
            payload[f"g0_{t}"] = model.g0[h]
            payload[f"mag_{t}"] = model.mag[h]
            payload[f"phase_{t}"] = model.phase[h]
            payload[f"rate_{t}"] = model.global_rate[h]
        np.savez(cache, **payload)
        print(f"  [fremen-fit] fit cached to {cache}")
    return _HBFremenAdapter(model, row_of), train_rate


def presence_evaluate(pipeline, graph, days, bounds, n_bins, grid_dt,
                      horizons=(5.0, 10.0), base_rates=None, minutes=0.0,
                      fremen=None, stream_open=None, hours=None,
                      hour_bins=False):
    """God's-eye presence: NO robot at test. Every mapped cell is scored on a
    `grid_dt` time grid across each held-out day, against the realised occupancy
    from ALL detections -- the same god's-eye test set as the flow metrics. The
    model is frozen (trained path-bounded via the robot); the test never uses the
    robot path. Each query is scored per event horizon in `horizons` (seconds):
    occupied within [t, t+h). Pools all days into one PresenceAccumulator."""
    import bisect
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from robot_sim import load_nav_graph
    from atc_stream import AtcStream
    VS = 0.4
    pos, _, _, _ = load_nav_graph(graph)
    # God's-eye: expose every mapped cell once, independent of any robot pose, so
    # presence_grid returns all covered cells at every query time.
    pipeline.set_visible_voxels(
        VS, [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(len(pos))])
    horizons = [float(h) for h in horizons]
    tags = [htag(h) for h in horizons]
    ests = ["kairos"]
    if fremen is not None:
        ests.append("fremen")
    acc = PresenceAccumulator(n_bins=n_bins, estimators=ests, horizons=horizons,
                              base_rates=base_rates)
    xb = (bounds[0], bounds[1]); yb = (bounds[2], bounds[3])
    for day in days:
        stream = (AtcStream(day, x_bounds=xb, y_bounds=yb) if stream_open is None
                  else stream_open(day, xb, yb))
        # Ground-truth per-voxel detection times (sorted) for occupancy lookup.
        vox = {}
        vx = np.floor(stream.x / VS).astype(np.int64)
        vy = np.floor(stream.y / VS).astype(np.int64)
        for i in range(stream.t.size):
            vox.setdefault((int(vx[i]), int(vy[i]), 0), []).append(float(stream.t[i]))
        # (stream is time-sorted, so each voxel's list is already sorted.)
        t0 = stream.t_min
        dur = (stream.t_max - t0) if minutes == 0 else minutes * 60.0
        n_steps = max(1, int(dur / grid_dt))
        tic = time.time(); nq = 0
        for k in range(n_steps):
            t = t0 + k * grid_dt
            if hours is not None:
                hh = (t % 86400.0) / 3600.0
                if not (hours[0] <= hh < hours[1]):
                    continue
            g = pipeline.presence_grid(t, horizons)
            gvx = np.asarray(g["vx"], dtype=np.int64); gvy = np.asarray(g["vy"], dtype=np.int64)
            gvz = np.asarray(g["vz"], dtype=np.int64)
            m = gvx.size
            if m == 0:
                continue
            y = {tag: np.zeros(m) for tag in tags}
            for i in range(m):
                times = vox.get((int(gvx[i]), int(gvy[i]), int(gvz[i])))
                if times is None:
                    continue
                lo = bisect.bisect_left(times, t)
                for h, tag in zip(horizons, tags):
                    y[tag][i] = 1.0 if bisect.bisect_left(times, t + h) > lo else 0.0
            preds = {"kairos": {tag: np.asarray(g[f"p{tag}"]) for tag in tags}}
            if fremen is not None:
                preds["fremen"] = {tag: fremen.predict(gvx, gvy, gvz, t, h)
                                   for h, tag in zip(horizons, tags)}
            acc.update(y, preds,
                       hour=int((t % 86400.0) // 3600.0) if hour_bins else None)
            nq += m
        print(f"  {pathlib.Path(day).name}: {nq} cell-times in {time.time()-tic:.0f}s")
    return acc.result()


def fit_fremen_occupancy(pipeline, graph, train_days, bounds, periods_s, grid_dt,
                         horizons=(5.0, 10.0), minutes=0.0):
    """Fit the FreMEn occupancy baseline: replay the robot coverage over the
    TRAINING days (same cell enumeration as the presence eval, via presence_grid)
    and accumulate per-cell (t, occupied) into a FremenOccupancy, one spectral
    model per (cell, horizon). Returns the finalized model and the per-horizon
    training occupancy rate. Uses the pipeline only to enumerate covered cells,
    not its presence prediction."""
    import bisect
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from robot_sim import load_nav_graph, build_coverage_loop, RobotSim, KEYFRAME_DT_S, ROBOT_SPEED_MPS
    from atc_stream import AtcStream
    VS = 0.4
    pos, _, adj, trav = load_nav_graph(graph)
    poly, _ = build_coverage_loop(pos, adj, trav)
    sim = RobotSim(poly, pos)
    horizons = [float(h) for h in horizons]
    fm = FremenOccupancy(periods_s, horizons=tuple(horizons))
    occ = {h: 0.0 for h in horizons}; occ_n = 0
    xb = (bounds[0], bounds[1]); yb = (bounds[2], bounds[3])
    for day in train_days:
        stream = AtcStream(day, x_bounds=xb, y_bounds=yb)
        vox = {}
        vx = np.floor(stream.x / VS).astype(np.int64); vy = np.floor(stream.y / VS).astype(np.int64)
        for i in range(stream.t.size):
            vox.setdefault((int(vx[i]), int(vy[i]), 0), []).append(float(stream.t[i]))
        t0 = stream.t_min
        dur = (stream.t_max - t0) if minutes == 0 else minutes * 60.0
        n_kf = int(dur / KEYFRAME_DT_S); next_grid_t = t0; tic = time.time(); nq = 0
        for k in range(n_kf):
            t = t0 + k * KEYFRAME_DT_S
            pose = sim.pose_at_arclength(ROBOT_SPEED_MPS * k * KEYFRAME_DT_S)
            pipeline.set_visible_voxels(VS, [[float(pos[i, 0]), float(pos[i, 1]), 0.0]
                                             for i in sim.visible_places(pose)])
            if t < next_grid_t:
                continue
            next_grid_t += grid_dt
            g = pipeline.presence_grid(t)
            gvx = np.asarray(g["vx"], np.int64); gvy = np.asarray(g["vy"], np.int64); gvz = np.asarray(g["vz"], np.int64)
            m = gvx.size
            if m == 0:
                continue
            y = {h: np.zeros(m) for h in horizons}
            for i in range(m):
                times = vox.get((int(gvx[i]), int(gvy[i]), int(gvz[i])))
                if times is None:
                    continue
                lo = bisect.bisect_left(times, t)
                for h in horizons:
                    y[h][i] = 1.0 if bisect.bisect_left(times, t + h) > lo else 0.0
            fm.fit_update(gvx, gvy, gvz, t, y)
            for h in horizons:
                occ[h] += float(y[h].sum())
            occ_n += int(m)
            nq += m
        print(f"  [fremen-fit] {pathlib.Path(day).name}: {nq} cell-times in {time.time()-tic:.0f}s")
    train_rate = {h: (occ[h] / occ_n if occ_n else None) for h in horizons}
    print("  [fremen-fit] TRAIN occupancy rate " +
          " ".join(f"{htag(h)}s={train_rate[h]}" for h in horizons))
    return fm.finalize(), train_rate


def _print_presence(pr):
    if pr is None:
        print("  (no presence rows)"); return
    order = ["base_rate", "fremen", "kairos"]
    for tag in pr:
        h = pr.get(tag)
        if h is None:
            continue
        print(f"  --- horizon {tag}s ---  occupancy_rate={h['occupancy_rate']:.4f}  "
              f"uncertainty(floor)={h['uncertainty']:.5f}  (n={h['n']})")
        print(f"    {'estimator':>15} {'BernLL':>9} {'reliab':>9} {'resol':>9}")
        for name in order:
            e = h.get(name)
            if e is None:
                continue
            tagc = "*" if name == "base_rate" and e.get("in_sample") else ""
            print(f"    {name+tagc:>15} {e['bern_ll']:>9.4f} "
                  f"{e['reliability']:>9.5f} {e['resolution']:>9.5f}")
        hr = h.get("hourly")
        if hr:
            ks = [k for k in range(24) if hr["n"][k]]
            print("    by hour of day (mean BernLL; occ = occupancy rate):")
            print("      " + " ".join(f"h{k:02d}".rjust(8) for k in ks))
            print("      " + " ".join(f"{hr['occ'][k]:8.4f}" for k in ks) + "  occ")
            for name in order:
                if hr.get(name) is None:
                    continue
                print("      " + " ".join(f"{hr[name][k]:8.4f}" for k in ks)
                      + f"  {name}")


# ---------------------------------------------------------------- online eval loop
class _RowDump:
    """Flat per-row heading-score dump in stream order (row i across the four
    files is detection i of the canonical stream, which is deterministic, so
    dumps from different methods pair row-for-row): <prefix>.good.u8,
    <prefix>.{lp,crps,mpp}.f32.bin, plus <prefix>.meta.json with the count."""

    def __init__(self, prefix):
        p = pathlib.Path(prefix)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._prefix = str(prefix)
        self._f = {"good": open(f"{prefix}.good.u8.bin", "wb"),
                   "lp": open(f"{prefix}.lp.f32.bin", "wb"),
                   "crps": open(f"{prefix}.crps.f32.bin", "wb"),
                   "mpp": open(f"{prefix}.mpp.f32.bin", "wb")}
        self._n = 0

    def write(self, good, lp, crps, mpp):
        self._f["good"].write(np.asarray(good, np.uint8).tobytes())
        self._f["lp"].write(np.asarray(lp, np.float32).tobytes())
        self._f["crps"].write(np.asarray(crps, np.float32).tobytes())
        self._f["mpp"].write(np.asarray(mpp, np.float32).tobytes())
        self._n += int(np.asarray(lp).size)

    def close(self):
        for f in self._f.values():
            f.close()
        pathlib.Path(self._prefix + ".meta.json").write_text(
            json.dumps({"n": self._n}) + "\n")


def online_evaluate(score_fn, source, n_bins, voxel_size=0.4,
                    coupling_json=None, coupling_window_s=1.0,
                    coverage_thresholds=(10, 50), lc_events_s=None, lc_window_s=5.0,
                    temporal_phases=None, temporal_elapsed_bins_min=(0, 15, 30, 60, 120),
                    speed_cap_mps=DEFAULT_SPEED_CAP_MPS, rho_max_override=0.0,
                    heading_only=False, dump_prefix=None, weight_cal=None):
    flow = FlowAccumulator(speed_cap_mps=speed_cap_mps, rho_max_override=rho_max_override)
    cal = CalibrationAccumulator(n_bins=n_bins)
    cov = CoverageStratifiedAccumulator(voxel_size=voxel_size, thresholds=coverage_thresholds)
    temporal = TemporalStratifiedAccumulator(bins_min=temporal_elapsed_bins_min)
    phase = PhaseStratifiedAccumulator(temporal_phases)
    # Only build the coupling accumulator when the method exports coupling edges
    # (baselines export none, so it is skipped for them).
    cpl = None
    if coupling_json:
        cpl = CouplingAccumulator(coupling_json, voxel_size=voxel_size,
                                  time_window_s=coupling_window_s)
        if not cpl.edge_lookup:
            cpl = None
    # Heading-distribution scores: mixture path (kairos slot weights, plus the
    # binned-sector control of the same forecast) or edge-aligned histogram
    # path (methods returning per-detection bin masses in `sector_masses`).
    hd = HeadingDistributionAccumulator()
    hd_hist = HistogramHeadingAccumulator(n_bins=8)
    hist_mode = None
    binned = BinnedControlAccumulator(n_bins=8)
    passages = PassageWeightedAccumulator(voxel_size=voxel_size)
    cellmean = CellMeanAccumulator(voxel_size=voxel_size)
    dump = _RowDump(dump_prefix) if dump_prefix else None
    HEADING_FLOOR = -math.log(2.0 * math.pi)
    # Loop-closure ΔMLPD: only when LC event timestamps are supplied.
    lc = LcMlpdDeltaAccumulator(lc_events_s, lc_window_s) if lc_events_s else None
    n = 0
    tic = time.time()
    for t_s, xyz, theta, rho, ids in source:
        m = len(t_s)
        # NB: densities are scored at the true observed speed for every method
        # (symmetric); the glitch-outlier cap is applied only inside FlowAccumulator
        # for rho_max / speed-MAE, so the shared MLPD floor stays physical.
        r = score_fn(t_s, xyz, theta, rho)
        flow.update(r["matched"], rho, r["log_p"], r["mean_rho"])
        if not heading_only:
            cal.update(r["matched"], rho, r["mean_rho"], r["std_rho"])
            cov.update(r["matched"], xyz, r["log_p"]["joint"])
            temporal.update(r["matched"], t_s, r["log_p"]["joint"])
            phase.update(r["matched"], t_s, r["log_p"]["joint"])
            if cpl is not None:
                cpl.update(r["matched"], t_s, xyz, theta, rho, r["slot_weights"], ids=ids)
            if lc is not None:
                lc.update(r["matched"], t_s, r["log_p"]["joint"])
            if weight_cal is not None:
                weight_cal.update(t_s, xyz, ids, r["matched"],
                                  r["responsibilities"], r["slot_weights"],
                                  r["slot_weight_stds"], r["slot_weight_clamped"],
                                  r["slot_weight_orders"], r["n_crossings"])
        sm = r.get("sector_masses")
        if hist_mode is None:
            hist_mode = sm is not None
        if hist_mode:
            crps_r, mpp_r, good = hd_hist.rows(r["matched"], theta, sm)
            hd_hist.accumulate(crps_r, mpp_r, good)
        else:
            crps_r, mpp_r, good, W = hd.rows(r["matched"], theta, r.get("slot_weights"))
            hd.accumulate(crps_r, mpp_r, good)
            binned.update(good, theta, hd.sector_masses(W), good)
        lp_h = np.asarray(r["log_p"]["heading"], float)
        lp_full = np.where(good & np.isfinite(lp_h), lp_h, HEADING_FLOOR)
        passages.update(t_s, xyz, ids, good, lp_full, crps_r, mpp_r)
        cellmean.update(xyz, good, lp_full, crps_r, mpp_r)
        if dump is not None:
            dump.write(good, lp_full, crps_r, mpp_r)
        n += m
        if n % 5_000_000 < m:
            print(f"  scored {n} ({n/(time.time()-tic):.0f}/s)")
    if dump is not None:
        dump.close()
    return {"flow": flow.result(), "cal": cal.result(), "cov": cov.result(),
            "heading_dist": (hd_hist if hist_mode else hd).result(),
            "binned_control": None if hist_mode else binned.result(),
            "passages": passages.result(),
            "cellmean": cellmean.result(),
            "coupling": cpl.result() if cpl else None,
            "temporal": temporal.result(), "phase": phase.result(),
            "weight_cal": weight_cal.result() if weight_cal else None,
            "lc": lc.result() if lc else None, "n": n}


def _print_flow(f):
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"
    print(f"  n={f['n_total']}  match_rate={f['match_rate']:.4f}  rho_max={f['rho_max']:.3f}")
    print(f"  MLPD heading : {fmt(f['mlpd_heading'])}  [matched-only {fmt(f['mlpd_heading_matched_only'])}]")
    print(f"  MLPD speed   : {fmt(f['mlpd_speed'])}  [matched-only {fmt(f['mlpd_speed_matched_only'])}]")
    print(f"  MLPD joint   : {fmt(f['mlpd_joint'])}  [matched-only {fmt(f['mlpd_joint_matched_only'])}]")
    if f.get("density_floor"):
        print(f"  below floor  : H {100*f['below_floor_heading']:.3f}%  "
              f"S {100*f['below_floor_speed']:.3f}%  J {100*f['below_floor_joint']:.3f}%  "
              f"of matched (density < {f['density_floor']:g})")
    print(f"  Speed MAE    : {fmt(f['speed_mae_mps'])}")


def _print_cal(c):
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"
    def row(name, a, n):
        if a is None:
            print(f"  {name:26s} --"); return
        print(f"  {name:26s} cov68={fmt(a['speed_coverage_68'])}  cov95={fmt(a['speed_coverage_95'])}  "
              f"ECE={fmt(a['speed_ece'])}  sharpness={fmt(a['speed_sharpness_mps'])}  (n={n})")
    row("predictive interval", {"speed_coverage_68": c["speed_coverage_68"],
                               "speed_coverage_95": c["speed_coverage_95"],
                               "speed_ece": c["speed_ece"],
                               "speed_sharpness_mps": c["speed_sharpness_mps"]},
        c["n_calibration"])


def _print_weight_cal(w):
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"
    for name, label in (("soft", "soft shares (primary)"),
                        ("hard", "argmax votes (robustness)")):
        v = w[name]
        print(f"  {label:26s} cov68={fmt(v['coverage_68'])}  cov95={fmt(v['coverage_95'])}  "
              f"ECE={fmt(v['ece'])}  sharpness={fmt(v['sharpness_share'])}  (n={v['n']})")
    bs = w.get("bootstrap")
    if bs:
        print(f"  cluster bootstrap ({bs['n_clusters']} voxel-day clusters): "
              f"cov68 [{bs['coverage_68_ci'][0]:.4f}, {bs['coverage_68_ci'][1]:.4f}]  "
              f"cov95 [{bs['coverage_95_ci'][0]:.4f}, {bs['coverage_95_ci'][1]:.4f}]  "
              f"ECE [{bs['ece_ci'][0]:.4f}, {bs['ece_ci'][1]:.4f}]")
    nul = w.get("ece_null")
    if nul:
        print(f"  ECE null (perfect model, this sample structure): "
              f"mean={nul['mean']:.4f}  p95={nul['p95']:.4f}")
    vb = w.get("voxel_balanced")
    if vb:
        print(f"  voxel-balanced: cov68={vb['coverage_68']:.4f}  "
              f"cov95={vb['coverage_95']:.4f}  ({vb['n_voxels']} voxels)")
    print(f"  matched vote fraction: {fmt(w.get('matched_vote_fraction'))}")
    c = w["counters"]
    print(f"  windows: {c['windows_scored']} scored, {c['windows_below_min']} below "
          f"min-count, {c['windows_truncated']} truncated; samples: "
          f"{c['samples_capped']} capped, {c['samples_clamped']} clamped, "
          f"{c['samples_order0']} order-0 of {c['samples_total']}")
    ps = w.get("parameter_share") or {}
    if ps:
        parts = [f"{lbl}: {fmt(d['mean'])} (n={d['n']})" for lbl, d in ps.items()]
        print("  parameter-variance share by training evidence: " + "  ".join(parts))
    for fam in ("clamp", "cap", "order", "burst", "count", "evidence", "slot"):
        d = w["strata"].get(fam) or {}
        parts = [f"{lbl}: cov95={fmt(t['coverage_95'])} (n={t['n']})"
                 for lbl, t in d.items()]
        if parts:
            print(f"  [{fam}] " + "  ".join(parts))


def _print_cov(cs):
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"
    lo, hi = cs.get("lo", 10), cs.get("hi", 50)
    print(f"  sparse(<{lo})={fmt(cs['mlpd_sparse'])} (n={cs['n_sparse']})  "
          f"medium({lo}-{hi})={fmt(cs['mlpd_medium'])} (n={cs['n_medium']})  "
          f"dense(>={hi})={fmt(cs['mlpd_dense'])} (n={cs['n_dense']})")


def _print_temporal(tp):
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"
    keys = [k for k in tp if k.startswith("temporal_mlpd_")]
    print(f"  session_start={tp['temporal_session_start_s']}  n_valid={tp['temporal_n_valid']}")
    for k in keys:
        span = k[len("temporal_mlpd_"):]
        print(f"    {span:>10}: MLPD {fmt(tp[k])}  (n={tp['temporal_n_' + span]})")


def _print_phase(ph):
    if not ph:
        print("  (no temporal phases configured)"); return
    def fmt(v):
        return "  --  " if v is None else f"{v:+.4f}"
    for name, d in ph.items():
        B = d["n_bins"]
        print(f"  [{name}] period={d['period_s']:.0f}s, {B} phase bins (MLPD | n):")
        # print bins compactly, a few per line
        for i in range(B):
            end = "\n" if (i % 6 == 5 or i == B - 1) else "  "
            print(f"    b{i:02d} {fmt(d['mlpd'][i])} n={d['n'][i]}", end=end)


def _print_lc(lc):
    if lc is None:
        print("  (no loop-closure events supplied)"); return
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"
    print(f"  mean ΔMLPD={fmt(lc['lc_mean_delta_mlpd'])}  frac_improved={fmt(lc['lc_frac_improved'])}  "
          f"events_evaluated={lc['lc_n_events_evaluated']}  W={lc['lc_window_s']}s")


def _print_heading_dist(hd):
    if hd is None:
        print("  (no detections)")
        return
    print(f"  circular CRPS (rad, lower better): {hd['crps_rad']:.4f}  "
          f"[matched-only {hd.get('crps_rad_matched_only', float('nan')):.4f}]  "
          f"(uniform floor {hd['crps_uniform']:.4f})")
    print(f"  MPP (mass on observed 45deg bin) : {hd['mpp']:.4f}  "
          f"[matched-only {hd.get('mpp_matched_only', float('nan')):.4f}]  "
          f"(uniform {hd['mpp_uniform']:.4f})")


def _print_passages(p):
    if p is None:
        print("  (no passages)")
        return
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"
    print(f"  n_passages={p['n_passages']:,}  (with matched rows: "
          f"{p.get('n_passages_matched', 0):,})")
    print(f"  MLPD heading : {fmt(p['mlpd_heading'])}  "
          f"[matched-only {fmt(p.get('mlpd_heading_matched_only'))}]")
    print(f"  circular CRPS: {fmt(p['crps_rad'])}  "
          f"[matched-only {fmt(p.get('crps_rad_matched_only'))}]  (uniform floor 0.7854)")
    print(f"  MPP          : {fmt(p['mpp'])}  "
          f"[matched-only {fmt(p.get('mpp_matched_only'))}]  (uniform 0.1250)")


def _print_cellmean(cm):
    if cm is None:
        print("  (no cells)")
        return
    lo, hi = cm["crps_ci95"]
    print(f"  n_cells={cm['n_cells']:,}  median det/cell={cm['median_det_per_cell']:.0f}")
    print(f"  MLPD heading : {cm['mlpd_heading']:.4f}   (per-cell equal weight)")
    print(f"  circular CRPS: {cm['crps_rad']:.4f}   "
          f"[bootstrap 95% CI over cells {lo:.4f}, {hi:.4f}]  (uniform floor 0.7854)")
    print(f"  MPP          : {cm['mpp']:.4f}   (uniform 0.1250)")
    print("  by min-detections/cell (robustness to rare-cell noise):")
    for t, v in sorted(cm["by_threshold"].items()):
        print(f"    >= {t:>3}: {v['n_cells']:>6,} cells  CRPS {v['crps_rad']:.4f}  "
              f"MLPD {v['mlpd_heading']:.4f}")


def _print_coupling(cp):
    def fmt(v, s="{:.4f}"):
        return "--" if v is None else s.format(v)
    if cp is None:
        print("  (no coupling edges exported)")
        return
    print(f"  n_pairs={cp['true_j_mlpd_n_pairs']}")
    print(f"  J-MLPD coupled : {fmt(cp['true_j_mlpd'])}  baseline {fmt(cp['true_j_mlpd_baseline_mlpd'])}  "
          f"gain {fmt(cp['true_j_mlpd_gain'])}  win {fmt(cp['j_mlpd_win_rate'])}")
    print(f"  Cond-heading MAE: coupled {fmt(cp['cond_heading_mae_deg'])}deg  "
          f"baseline {fmt(cp['cond_heading_mae_baseline_deg'])}deg  "
          f"gain {fmt(cp['cond_heading_mae_gain_deg'])}deg  win {fmt(cp['cond_heading_win_rate'])}")
    ag = cp["agent_strata"]
    for nm in ("same", "cross"):
        a = ag[nm]
        print(f"  {nm}-agent: J-MLPD gain {fmt(a['j_mlpd_gain'])}  "
              f"cond-MAE gain {fmt(a['cond_heading_mae_gain_deg'])}deg  (n={a['n_pairs']})")
    if ag["n_pairs_id_unknown"]:
        print(f"  ({ag['n_pairs_id_unknown']} pairs lacked track ids)")
    ds = cp["disagree_subset"]
    print(f"  disagree subset (obs headings >45deg): J-MLPD gain {fmt(ds['j_mlpd_gain'])}  "
          f"cond-MAE gain {fmt(ds['cond_mae_gain_deg'])}deg  (n={ds['n_pairs']})")
    for s in ("low", "medium", "high"):
        k = cp["kl_strata"][s]
        print(f"  KL {s:<6}: J-MLPD gain {fmt(k['j_mlpd_gain'])}  "
              f"cond-MAE gain {fmt(k['cond_mae_gain_deg'])}deg  (n={k['n_pairs']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="kairos",
                    help="Method to score. Names an adapter in baselines/; "
                         "available: " + ", ".join(baselines.available()))
    ap.add_argument("--dataset", default="atc", choices=["atc", "tbd"])
    ap.add_argument("--tbd-tracks", nargs="+", default=None,
                    help="held-out TBD scene tracks.csv (god's-eye: all detections)")
    ap.add_argument("--config-name", default="atc")
    ap.add_argument("--labelspace", default="ade20k_outdoor")
    ap.add_argument("--flow-state", default=None)
    ap.add_argument("--scene-dir", default=None)
    ap.add_argument("--graph", default=None,
                    help="authored graph the model was trained on; for kairos its "
                         "navigational layer is installed so evidence sharing acts")
    ap.add_argument("--days", nargs="+", default=None, help="held-out test-day npz/csv")
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--horizon-s", type=float, default=0.0)
    ap.add_argument("--coupling-json", default=None,
                    help="EdgeCouplingMap JSON for the coupling metric; if omitted "
                         "the kairos path exports it from the loaded model")
    ap.add_argument("--no-coupling", action="store_true",
                    help="skip the coupling metric even if the model exports edges")
    ap.add_argument("--coupling-window-s", type=float, default=1.0)
    ap.add_argument("--coupling-voxel-size-m", type=float, default=0.4)
    ap.add_argument("--coverage-thresholds", type=int, nargs=2, default=[10, 50],
                    metavar=("LO", "HI"),
                    help="sparse<LO / LO<=medium<HI / dense>=HI per-voxel detection "
                         "counts (TBD: 10 50; ATC dense terciles: 5000 20000)")
    ap.add_argument("--lc-events-json", default=None,
                    help="JSON list of loop-closure event timestamps (s) for the "
                         "loop-closure ΔMLPD; omit to skip it")
    ap.add_argument("--lc-window-s", type=float, default=5.0,
                    help="pre/post window (s) around each loop-closure event")
    ap.add_argument("--temporal-phases", default="daily:86400:24 weekly:604800:7",
                    help="phase-fold periods for the time-of-day MLPD, per dataset "
                         "(like the NUDFT omegas): 'name:period_s:n_bins ...'. "
                         "Empty string disables. Default = ATC daily(hour-of-day)+weekly(day-of-week)")
    ap.add_argument("--temporal-elapsed-bins-min", default="0 15 30 60 120",
                    help="session-elapsed bin edges in minutes for the TBD-style "
                         "temporal stratification")
    ap.add_argument("--presence", action="store_true",
                    help="run the presence (occupancy) evaluation via robot-coverage "
                         "replay instead of the per-detection flow metrics")
    ap.add_argument("--presence-grid-dt", type=float, default=1.0)
    ap.add_argument("--presence-hourly", action="store_true",
                    help="additionally report presence metrics folded by hour "
                         "of day (24 bins), every estimator")
    ap.add_argument("--fremen-select", action="store_true",
                    help="hb fit only: prequential per-cell selection of the "
                         "number of spectral components (the FreMEn model-order "
                         "choice, leak-free), mirroring the kairos gate")
    ap.add_argument("--fremen-cache", default=None,
                    help="npz path for the hb FreMEn fit: loaded if present "
                         "(skipping the training pass), written after a fresh "
                         "fit; ignored when --presence-hours is set")
    ap.add_argument("--presence-hours", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="restrict presence scoring to query times whose local "
                         "hour lies in [LO, HI); the constant base rate is "
                         "recomputed on the same restricted training windows")
    ap.add_argument("--presence-horizons", type=float, nargs="+",
                    default=[5.0, 10.0],
                    help="presence event horizons Delta_t in seconds: a query at "
                         "time t scores occupancy within [t, t+Delta_t); one "
                         "result block per horizon")
    ap.add_argument("--stream-source", default="atc", choices=["atc", "hb"],
                    help="presence-replay stream: 'hb' reads station dates "
                         "(YYYY-MM-DD in --days/--fremen-train-days) via "
                         "HBDayStream (per-track 10 Hz interpolation) and fits "
                         "FreMEn full-visibility (no robot exists at the site)")
    ap.add_argument("--hb-root", default="/data/hb-station/Data-2")
    ap.add_argument("--station", default="HB", choices=["HB", "PB"])
    ap.add_argument("--interp-hz", type=float, default=10.0)
    ap.add_argument("--gap-max-s", type=float, default=2.0)
    ap.add_argument("--fremen-train-days", nargs="+", default=None,
                    help="train-day npz for the FreMEn occupancy baseline; fit on "
                         "these, then score it as a presence baseline on --days")
    ap.add_argument("--fremen-periods-s", type=float, nargs="+",
                    default=[3600.0, 43200.0, 86400.0, 604800.0],
                    help="FreMEn candidate periods (dataset parameter, like the omegas)")
    ap.add_argument("--base-rate-5s", type=float, default=None,
                    help="training-set occupancy rate (5 s) for the base-rate baseline")
    ap.add_argument("--base-rate-10s", type=float, default=None)
    ap.add_argument("--minutes", type=float, default=0.0, help="cap replay/scoring minutes (0=full)")
    ap.add_argument("--rho-max", type=float, default=0.0,
                    help="pin the MLPD floor domain to this speed (m/s) instead of "
                         "the running max over the streamed test set; use one value "
                         "across every method being compared (0 = running max)")
    ap.add_argument("--map-pgm", default=None,
                    help="ATC occupancy PGM; with --map-yaml, score on the canonical "
                         "map-extent row set instead of the nav-graph box")
    ap.add_argument("--map-yaml", default=None)
    ap.add_argument("--dump-prefix", default=None,
                    help="write flat per-row heading scores (good/lp/crps/mpp) under "
                         "this path prefix, in stream order, for paired analysis")
    ap.add_argument("--cellmap-out", default=None,
                    help="write per-cell heading scores (vx/vy/crps/mlpd/mpp/n) to this "
                         ".npz for the spatial heatmap; no re-run needed to build it")
    ap.add_argument("--speed-cap-mps", type=float, default=3.0,
                    help="clip observed speeds to this physical pedestrian max (m/s) "
                         "before scoring and rho_max; use the SAME value across methods "
                         "(0 disables)")
    ap.add_argument("--weight-calibration", action="store_true",
                    help="score mixing-weight calibration (beta-binomial coverage of "
                         "windowed responsibility shares). kairos only")
    ap.add_argument("--weight-dt-win", type=float, default=None,
                    help="weight-calibration window width (s); a dataset-scale "
                         "parameter like the NUDFT omegas, required with "
                         "--weight-calibration")
    ap.add_argument("--weight-min-count", type=int, default=5,
                    help="minimum matched crossings per window to score it")
    ap.add_argument("--weight-visit-gap-s", type=float, default=2.0,
                    help="crossing closure gap (s); MUST equal the training "
                         "config's visit_gap_s so eval-side crossings are the "
                         "trainer's vote unit")
    ap.add_argument("--weight-visit-max-s", type=float, default=10.0,
                    help="crossing span cap (s); MUST equal the training config's "
                         "visit_max_s")
    ap.add_argument("--weight-seed", type=int, default=0,
                    help="seed for the randomized PIT and the cluster bootstrap")
    # Each adapter registers its own options, so a method can take settings the
    # evaluator knows nothing about.
    for _name in baselines.available():
        baselines.resolve(_name).add_arguments(ap)

    args = ap.parse_args()

    if args.method == "kairos":
        baselines.resolve("kairos").require_graph(args)

    if args.weight_calibration:
        if args.method != "kairos":
            raise SystemExit("[evaluate] --weight-calibration needs --method kairos "
                             "(the score API's weight channels)")
        if args.weight_dt_win is None:
            raise SystemExit("[evaluate] --weight-calibration requires --weight-dt-win "
                             "(dataset-scale window width in seconds)")
        if args.horizon_s != 0.0:
            raise SystemExit("[evaluate] --weight-calibration assumes horizon 0 "
                             "(windows are centred on the time the interval is "
                             "issued for)")

    if args.dataset == "atc":
        if args.map_pgm and args.map_yaml:
            sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
            from atc_stream import map_bounds
            xb, yb = map_bounds(args.map_pgm, args.map_yaml)
            bounds = (xb[0], xb[1], yb[0], yb[1])
            print(f"[evaluate] canonical map-extent bounds: "
                  f"x=({bounds[0]:.2f}, {bounds[1]:.2f}) y=({bounds[2]:.2f}, {bounds[3]:.2f})")
        else:
            bounds = _graph_bounds(args.graph)
    else:
        bounds = None

    # The training-stream opener, when the dataset needs one. Adapters that fit
    # at construction time read it off the Context; the presence replay below
    # uses the same object.
    stream_open = None
    if args.stream_source == "hb":
        stream_open = hb_stream_opener(args.hb_root, args.station,
                                       args.interp_hz, args.gap_max_s)

    # The method seam: --method names an adapter in baselines/, which loads or
    # fits its model and hands back a score_fn. Every method from here on runs
    # through the same loop and the same accumulators.
    ctx = baselines.Context(bounds=bounds, dataset=args.dataset,
                            stream_open=stream_open, horizon_s=args.horizon_s)
    scorer = baselines.resolve(args.method).make_scorer(args, ctx)
    score_fn = scorer.score_fn
    heading_only = scorer.heading_only
    coupling_json = scorer.coupling_json
    pipeline = scorer.handle

    # The temporal stratifications assume the test stream arrives in time order;
    # day files are named by date, so lexicographic order == chronological.
    day_list = args.days if args.dataset == "atc" else (args.tbd_tracks or [])
    if day_list:
        names = [pathlib.Path(d).name for d in day_list]
        if names != sorted(names):
            raise SystemExit("[evaluate] --days/--tbd-tracks must be passed in "
                             "chronological (sorted) order; got: " + " ".join(names))

    if args.presence and pipeline is None:
        raise SystemExit("[evaluate] --presence needs a method that exposes a "
                         "pipeline handle (currently --method kairos)")

    if args.presence:
        if args.stream_source == "hb":
            print("===== ONLINE PRESENCE (full-visibility station replay) =====")
        else:
            print("===== ONLINE PRESENCE (robot-coverage replay) =====")
        horizons = [float(h) for h in args.presence_horizons]
        fremen = None
        base_rates = {}
        if args.base_rate_5s is not None:
            base_rates[5.0] = args.base_rate_5s
        if args.base_rate_10s is not None:
            base_rates[10.0] = args.base_rate_10s
        if args.fremen_train_days:
            print("----- fitting FreMEn occupancy baseline on training days -----")
            if args.stream_source == "hb":
                fremen, train_rate = fit_fremen_hb(args.graph,
                                          args.fremen_train_days, stream_open,
                                          bounds, args.fremen_periods_s,
                                          args.presence_grid_dt, horizons=horizons,
                                          hours=args.presence_hours,
                                          cache=args.fremen_cache,
                                          select=args.fremen_select)
            else:
                fremen, train_rate = fit_fremen_occupancy(pipeline, args.graph,
                                          args.fremen_train_days,
                                          bounds, args.fremen_periods_s,
                                          args.presence_grid_dt, horizons=horizons,
                                          minutes=args.minutes)
            for h in horizons:
                base_rates.setdefault(h, train_rate[h])
            print("[base-rate] using TRAIN occupancy rate " +
                  " ".join(f"{htag(h)}s={base_rates[h]}" for h in horizons))
        pr = presence_evaluate(pipeline, args.graph, args.days, bounds, args.n_bins,
                               args.presence_grid_dt, horizons=horizons,
                               base_rates=base_rates, minutes=args.minutes,
                               fremen=fremen, stream_open=stream_open,
                               hours=args.presence_hours,
                               hour_bins=args.presence_hourly)
        print("----- PRESENCE (Bernoulli-LL / reliability / resolution) -----"); _print_presence(pr)
        return

    if args.dataset == "atc" and args.stream_source == "hb":
        source = hb_source(args.hb_root, args.station, args.days, bounds,
                              args.chunk, args.interp_hz,
                              args.gap_max_s, minutes=args.minutes)
    elif args.dataset == "atc":
        source = atc_source(args.days, bounds, args.chunk, minutes=args.minutes)
    elif args.dataset == "tbd":
        source = tbd_source(args.tbd_tracks, args.chunk, minutes=args.minutes)
    else:
        raise NotImplementedError(args.dataset)

    lc_events = None
    if args.lc_events_json:
        import json as _json
        lc_events = _json.load(open(args.lc_events_json))
    # Parse "name:period_s:n_bins ..." into the phase list (period a dataset param).
    phases = []
    for tok in args.temporal_phases.split():
        nm, per, nb = tok.split(":")
        phases.append((nm, float(per), int(nb)))
    elapsed_bins = tuple(int(x) for x in args.temporal_elapsed_bins_min.split())

    wcal = None
    if args.weight_calibration:
        wcal = WeightCalibrationAccumulator(
            dt_win=args.weight_dt_win, min_count=args.weight_min_count,
            voxel_size=args.coupling_voxel_size_m,
            visit_gap_s=args.weight_visit_gap_s,
            visit_max_s=args.weight_visit_max_s,
            seed=args.weight_seed)

    print("===== ONLINE EVALUATE =====")
    res = online_evaluate(score_fn, source, args.n_bins,
                          voxel_size=args.coupling_voxel_size_m,
                          coupling_json=coupling_json,
                          coupling_window_s=args.coupling_window_s,
                          coverage_thresholds=tuple(args.coverage_thresholds),
                          lc_events_s=lc_events, lc_window_s=args.lc_window_s,
                          temporal_phases=phases, temporal_elapsed_bins_min=elapsed_bins,
                          speed_cap_mps=args.speed_cap_mps,
                          rho_max_override=args.rho_max,
                          heading_only=heading_only,
                          dump_prefix=args.dump_prefix, weight_cal=wcal)
    print("\n----- FLOW -----"); _print_flow(res["flow"])
    print("----- HEADING DISTRIBUTION (circular CRPS / MPP) -----"); _print_heading_dist(res["heading_dist"])
    if res.get("binned_control"):
        print("----- BINNED-MIXTURE CONTROL (same forecast, mass per shared sector) -----")
        _print_heading_dist(res["binned_control"])
    print("----- PER-PASSAGE HEADING (one vote per track-voxel crossing) -----")
    _print_passages(res["passages"])
    print("----- PER-CELL HEADING (equal weight per cell; exploration metric) -----")
    _print_cellmean(res["cellmean"])
    cm = res["cellmean"]
    if args.cellmap_out and cm is not None:
        np.savez(args.cellmap_out, vx=cm["cell_vx"], vy=cm["cell_vy"],
                 crps=cm["per_cell_crps"], mlpd=cm["per_cell_mlpd"],
                 mpp=cm["per_cell_mpp"], n=cm["per_cell_n"],
                 voxel_size=np.float64(args.coupling_voxel_size_m))
        print(f"[evaluate] per-cell heatmap arrays -> {args.cellmap_out}")
    if not heading_only:
        print("----- UNCERTAINTY -----"); _print_cal(res["cal"])
        if res.get("weight_cal"):
            print("----- WEIGHT CALIBRATION (beta-binomial share coverage) -----")
            _print_weight_cal(res["weight_cal"])
        print("----- COVERAGE-STRATIFIED MLPD (joint) -----"); _print_cov(res["cov"])
        print("----- COUPLING (adjacent-voxel pairs, KL-stratified) -----"); _print_coupling(res["coupling"])
        print("----- TEMPORAL PHASE-STRATIFIED MLPD (by period phase) -----"); _print_phase(res["phase"])
        print("----- TEMPORAL-STRATIFIED MLPD (elapsed since session start) -----"); _print_temporal(res["temporal"])
        print("----- LOOP-CLOSURE ΔMLPD -----"); _print_lc(res["lc"])


if __name__ == "__main__":
    main()
