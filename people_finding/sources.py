"""Per-system activity tables for the people-finding evaluation.

Every information source is reduced to the same object: an activity table
A[day, slot, node] over the authored graph's traversable nodes, slot length
SLOT_S, one row set of test days. The planner consumes these tables and
nothing else, so the decision policy is identical across systems.

Host-computable sources (this script):
  train_marginals  -> A_static.npz (per-cell training-mean rate, no time)
  fremen           -> A_fremen.npz (gated FreMEn from the selection cache,
                      full-visibility fit, its only regime)

Kairos's own source, lambda(t) via presence_grid, is produced by
export_kairos_activity.py, which runs in the pipeline container.

    python3 sources.py --dataset hb --source train_marginals
    python3 sources.py --dataset hb --source fremen
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
sys.path.insert(0, str(_HERE.parent / "evaluation"))
sys.path.insert(0, str(_HERE))

from planner import VOXEL, load_graph  # noqa: E402

import os

# Roots are environment-configurable so the tables can be built anywhere.
DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", pathlib.Path.home() / "datasets"))
R = pathlib.Path(os.environ.get("KAIROS_RESULTS", _HERE.parents[0] / "results"))

def _graph(shipped, authored):
    """The navigation graph shipped with the data, else the one
    grounded_sim/train_test.sh authored into the results folder."""
    return str(shipped if shipped.exists() else authored)


ATC_TEST = ["20130901", "20130904", "20130911", "20130915",
            "20130918", "20130922", "20130925", "20130929"]

DATASETS = {
    "hb": dict(
        kind="hb",
        graph=_graph(DATA_ROOT / "hb-station/graph/hb_flow_graph.json",
                     R / "hb/graph/hb_flow_graph.json"),
        root=str(DATA_ROOT / "hb-station/Data-2"),
        station="HB",
        train=("2022-09-05", "2023-04-23"),
        test=("2023-04-24", "2023-05-21"),
        bounds=(-2.0, 35.0, -32.0, 2.0),
        tz_s=7200.0,
        fremen_cache=None,                   # fitted on first use, cached under out
        out=str(R / "people_finding" / "hb"),
    ),
    "atc": dict(
        kind="atc",
        graph=_graph(DATA_ROOT / "atc/graph/atc_graph.json",
                     R / "atc/graph/atc_graph.json"),
        npz_dir=str(DATA_ROOT / "atc/npz"),
        test_days=ATC_TEST,
        bounds=(-60.0, 80.0, -40.0, 20.0),   # canonical map extent
        tz_s=32400.0,                        # JST
        fremen_cache=None,                   # fitted on first use, cached under out
        out=str(R / "people_finding" / "atc"),
    ),
}
SLOT_S = 600.0
N_SLOTS = int(86400 / SLOT_S)
HORIZON_S = 5.0   # rate-like: p(5s) ~ lambda*5s, unsaturated
PERIODS_S = [3600.0, 43200.0, 86400.0, 604800.0]


def hb_train_test_days(cfg):
    if cfg["kind"] == "atc":
        import glob
        import os
        days = sorted(os.path.basename(f)[4:-4] for f in
                      glob.glob(f"{cfg['npz_dir']}/atc-*.npz"))
        train = [d for d in days if d not in cfg["test_days"]]
        return train, list(cfg["test_days"])
    from hb_stream import station_dates
    d = station_dates(cfg["root"], cfg["station"])
    train = [x for x in d if cfg["train"][0] <= x <= cfg["train"][1]]
    test = [x for x in d if cfg["test"][0] <= x <= cfg["test"][1]]
    return train, test


def day0_epoch(day_iso):
    from datetime import date
    s = day_iso.replace("-", "")
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    return (date(y, m, d) - date(1970, 1, 1)).days * 86400.0


def open_day(cfg, day):
    xb = (cfg["bounds"][0], cfg["bounds"][1])
    yb = (cfg["bounds"][2], cfg["bounds"][3])
    if cfg["kind"] == "atc":
        from atc_stream import AtcStream
        return AtcStream(f"{cfg['npz_dir']}/atc-{day}.npz",
                         x_bounds=xb, y_bounds=yb)
    from hb_stream import HBDayStream, hb_day_files
    return HBDayStream(hb_day_files(cfg["root"], cfg["station"], day),
                          x_bounds=xb, y_bounds=yb,
                          interp_hz=10.0, gap_max_s=2.0)


def node_cells(pos):
    return (np.floor(pos[:, 0] / VOXEL).astype(np.int64),
            np.floor(pos[:, 1] / VOXEL).astype(np.int64))


def save(cfg, name, A, days, extra=None):
    out = pathlib.Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"A_{name}.npz", A=A.astype(np.float32),
                        days=np.array(days), slot_s=SLOT_S,
                        graph=cfg["graph"], **(extra or {}))
    print(f"[{name}] wrote {out}/A_{name}.npz  A{A.shape} "
          f"mean {A.mean():.4g}")


def build_train_marginals(cfg):
    """One pass over the training days: per-node detection rate (static)."""
    pos, ids, src, dst = load_graph(cfg["graph"])
    kvx, kvy = node_cells(pos)
    cell_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(kvx, kvy))}
    train, test = hb_train_test_days(cfg)
    counts = np.zeros(len(ids))
    tic = time.time()
    for di, day in enumerate(train):
        s = open_day(cfg, day)
        ix = np.floor(np.asarray(s.x) / VOXEL).astype(np.int64)
        iy = np.floor(np.asarray(s.y) / VOXEL).astype(np.int64)
        rows = np.array([cell_of.get((a, b), -1) for a, b in zip(ix, iy)])
        seen = rows >= 0
        np.add.at(counts, rows[seen], 1.0)
        if (di + 1) % 40 == 0:
            print(f"  [train] {di+1}/{len(train)} days "
                  f"({(time.time()-tic)/60:.1f} min)", flush=True)
    rate = counts / (len(train) * 86400.0)          # detections per second
    A_static = np.tile(rate[None, None, :], (len(test), N_SLOTS, 1))
    save(cfg, "static", A_static, test)


def fit_fremen(cfg, cache_path, grid_dt=5):
    """Fit the gated (prequentially selected) FreMEn occupancy model on the
    dataset's training days over the authored graph's cell universe, exactly as
    the station fit does (evaluate.py fit_fremen_hb), and cache it in
    the same npz schema. Full visibility by construction."""
    import json
    from hb_presence import SpectralOccupancy, occupied_runs
    doc = json.load(open(cfg["graph"]))
    all_nodes = [n for n in doc["nodes"] if int(n["layer"]) == 3]
    apos = np.array([n["attributes"]["position"][:2] for n in all_nodes])
    kvx = np.floor(apos[:, 0] / VOXEL).astype(np.int64)
    kvy = np.floor(apos[:, 1] / VOXEL).astype(np.int64)
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(kvx, kvy))}
    h = HORIZON_S
    model = SpectralOccupancy(len(row_of), PERIODS_S, [h], grid_dt)
    model.enable_selection()
    train, _ = hb_train_test_days(cfg)
    tic = time.time()
    for di, day in enumerate(train):
        s = open_day(cfg, day)
        vx = np.floor(s.x / VOXEL).astype(np.int64)
        vy = np.floor(s.y / VOXEL).astype(np.int64)
        rows = np.array([row_of.get((int(a), int(b)), -1)
                         for a, b in zip(vx, vy)], dtype=np.int64)
        m = rows >= 0
        t, rows = s.t[m], rows[m]
        o = np.lexsort((t, rows))
        t, rows = t[o], rows[o]
        day0 = int(day0_epoch(day))
        runs_by_h = {h: occupied_runs(rows, t, day0, h, grid_dt)}
        model.score_day_prequential(runs_by_h, day0)
        model.add_day(runs_by_h, day0)
        if (di + 1) % 10 == 0:
            print(f"  [fremen-fit] {di+1}/{len(train)} days "
                  f"({(time.time()-tic)/60:.1f} min)", flush=True)
    model.finalize()
    model.finalize_selection()
    np.savez_compressed(
        cache_path,
        g0_5=model.g0[h], mag_5=model.mag[h], phase_5=model.phase[h],
        rate_5=model.global_rate[h], horizons=[h], periods_s=PERIODS_S,
        grid_dt=grid_dt, n_cells=len(row_of), selected=1)
    print(f"[fremen-fit] cached to {cache_path}")


def build_fremen(cfg):
    """Gated FreMEn from the selection cache: p(occupied within HORIZON_S)
    per node at each slot start. Full-visibility fit by construction."""
    from hb_presence import SpectralOccupancy
    if cfg["fremen_cache"] is None:
        cfg["fremen_cache"] = str(pathlib.Path(cfg["out"]) / "fremen_fit_sel.npz")
    if not pathlib.Path(cfg["fremen_cache"]).exists():
        pathlib.Path(cfg["out"]).mkdir(parents=True, exist_ok=True)
        fit_fremen(cfg, cfg["fremen_cache"])
    z = np.load(cfg["fremen_cache"])
    assert int(z["selected"]) == 1, "expected the selected (gated) fit"
    assert HORIZON_S in [float(x) for x in z["horizons"]]
    h = HORIZON_S
    model = SpectralOccupancy(int(z["n_cells"]), PERIODS_S,
                              [h], int(z["grid_dt"]))
    model.g0 = {h: z["g0_5"]}
    model.mag = {h: z["mag_5"]}
    model.phase = {h: z["phase_5"]}
    model.global_rate = {h: float(z["rate_5"])}

    # cache rows are keyed by floor(pos/VOXEL) over ALL layer-3 nodes
    import json
    doc = json.load(open(cfg["graph"]))
    all_nodes = [n for n in doc["nodes"] if int(n["layer"]) == 3]
    apos = np.array([n["attributes"]["position"][:2] for n in all_nodes])
    row_of = {(int(np.floor(p[0] / VOXEL)), int(np.floor(p[1] / VOXEL))): i
              for i, p in enumerate(apos)}
    assert len(row_of) == int(z["n_cells"]), (len(row_of), int(z["n_cells"]))

    pos, ids, src, dst = load_graph(cfg["graph"])
    kvx, kvy = node_cells(pos)
    rows = np.array([row_of[(int(a), int(b))] for a, b in zip(kvx, kvy)])
    train, test = hb_train_test_days(cfg)
    A = np.zeros((len(test), N_SLOTS, len(ids)))
    for di, day in enumerate(test):
        t0 = day0_epoch(day)
        for si in range(N_SLOTS):
            A[di, si] = model.predict_at(rows, t0 + si * SLOT_S, h)
    save(cfg, "fremen", A, test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hb", choices=list(DATASETS))
    ap.add_argument("--source", required=True,
                    choices=["train_marginals", "fremen"])
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    if args.source == "train_marginals":
        build_train_marginals(cfg)
    elif args.source == "fremen":
        build_fremen(cfg)


if __name__ == "__main__":
    main()
