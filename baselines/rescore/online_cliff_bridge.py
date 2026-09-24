#!/usr/bin/env python3
"""Scoring bridge for the Fast Online CLiFF-map baseline.

The online CLiFF-map implementation (set ONLINE_MOD_DIR to its checkout)
is a map of dynamics that fits, per spatial cell, a semi-wrapped Gaussian mixture
(SWGMM) over (speed, heading) and updates it online as new observation batches
arrive. This module is the benchmark-side glue that lets it be scored under the
exact same protocol as the batch CLiFF-map baseline
(`rescore_heading_dist_cliff_tbd.py`): only the map construction changes (the
online SWGMM instead of `cliffmap.grid.CliffMapGrid`); the per-cell heading
marginal on a 720-point grid and the `online_metrics.GridHeadingScorer` with the
shared uniform-heading floor are identical.

Nothing in that checkout is modified: this file imports the online
updater as a library, drives it batch by batch, reads the SWGMM snapshot CSV it
writes, and reconstructs each cell's heading density.

The online CLiFF-map is static in time (no temporal covariate), so it produces
one heading density per cell regardless of query time, exactly like the batch
CLiFF-map baseline.

Runs in that method's own venv; needs no ROS or spark_dsg.
"""
import collections
import math
import os
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

_HERE = pathlib.Path(__file__).resolve().parent
ONLINE_MOD_DIR = os.environ.get("ONLINE_MOD_DIR", "online-mod")
sys.path.insert(0, str(_HERE))
sys.path.insert(0, ONLINE_MOD_DIR)

from online_metrics import GridHeadingScorer  # noqa: E402

G_GRID = 720
THETA_G = np.arange(G_GRID) * (2.0 * math.pi / G_GRID)

# Online CLiFF-map published configuration (online-mod/config_atc.yaml). The grid
# resolution (radius/step) is the method's own published 1 m; expose it so a
# resolution-matched comparison against the 0.4 m batch baselines can be run, but
# the default reproduces the published config.
DEFAULT_PARAMS = {
    "map_file": "maps/localization_grid_white.jpg",
    "raw_dataset": "OBSERVE",
    "radius": 1.0,
    "step": 1.0,
    "dataset_type": "OBSERVE",
    "convergence_thres_ms": 1e-5,
    "group_distance_tolerance": 1,
    "cluster_all": True,
    "too_few_data_thres": 5,
    "max_iter_ms": 100,
    "wind_num": 1,
    "convergence_thres_em": 1e-5,
    "max_iter_em": 100,
    "decay_rate": 0.5,
    "combine_thres": 0.1,
}


# One SWGMM component of a cell, named so the (speed, heading) ordering the
# online updater writes cannot be confused with cliffmap's (heading, speed).
Comp = collections.namedtuple("Comp", "w mean_h var_h mean_s var_s cov_sh")

# Matches cliffmap.grid.CliffMapGrid.score_row: a density of zero becomes
# log(1e-300) rather than -inf, so one unreachable observation cannot swallow
# the whole mean.
LOG_FLOOR = math.log(1e-300)


def detections_to_online_csv(t_s, x, y, rho, theta, out_csv, person_id=None):
    """Adapter: write detections in the online CLiFF-map 6-column CSV schema
    (time, person_id, x, y, velocity, motion_angle) -- the format its own
    DataLoader emits for the ATC/MAPF datasets. `theta` is the heading (rad) and
    `rho` the speed (m/s). person_id is carried for schema fidelity only; the
    builder's OBSERVE path uses only (x, y, velocity, motion_angle)."""
    n = len(t_s)
    pid = person_id if person_id is not None else np.zeros(n, dtype=np.int64)
    df = pd.DataFrame({
        "time": np.asarray(t_s, dtype=float),
        "person_id": np.asarray(pid),
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "velocity": np.asarray(rho, dtype=float),
        "motion_angle": np.asarray(theta, dtype=float),
    })
    pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, header=False, index=False)
    return out_csv


def _load_online_csv_as_batch(csv_path):
    """Load a 6-column online CLiFF-map CSV as the DataFrame the OBSERVE builder
    expects (columns time, x, y, velocity, motion_angle)."""
    df = pd.read_csv(csv_path, header=None,
                     names=["time", "person_id", "x", "y", "velocity", "motion_angle"])
    return df[["time", "x", "y", "velocity", "motion_angle"]]


class OnlineCliffModel:
    """Drive the online CLiFF-map updater over a sequence of observation batches
    and expose each cell's heading marginal for scoring.

    Each training batch is one call to `OnlineUpdateMoD.updateMoD`, which writes
    a full SWGMM snapshot (every cell's components) to its own CSV. The final
    batch's snapshot is the trained map; `build_scorer` reconstructs the
    720-point heading density per cell from it.
    """

    def __init__(self, params=None, work_dir=None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.step = float(self.params["step"])
        self._own_work = work_dir is None
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="online_cliff_")
        os.makedirs(self.work_dir, exist_ok=True)
        self.cliff_folder = os.path.join(self.work_dir, "cliffmaps")
        self.fig_folder = os.path.join(self.work_dir, "figs")
        self.config_path = os.path.join(self.work_dir, "config.yaml")
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.params, f)
        self._final_snapshot = None
        self.cell_dens = {}

    def train(self, batch_csvs, verbose=True):
        """Fit the online map by feeding each CSV as one online update batch, in
        order. `batch_csvs` is the list produced by the adapter (per scene / per
        day). Returns the final snapshot CSV path."""
        from mod_online_update import OnlineUpdateMoD  # online-mod, unmodified
        driver = OnlineUpdateMoD(
            config_file=self.config_path,
            current_cliff=None,
            output_cliff_folder=self.cliff_folder,
            save_fig_folder=self.fig_folder,
        )
        last = None
        for i, csv_path in enumerate(batch_csvs):
            batch = _load_online_csv_as_batch(csv_path)
            tag = f"batch_{i:04d}"
            if verbose:
                print(f"[online-cliff] batch {i+1}/{len(batch_csvs)}: "
                      f"{len(batch)} dets from {pathlib.Path(csv_path).name}",
                      flush=True)
            driver.updateMoD(batch, tag)
            last = os.path.join(self.cliff_folder, f"{tag}.csv")
        self._final_snapshot = last
        return last

    def train_lazy(self, produce, n_batches, verbose=True, cleanup_batches=True):
        """Fit the online map when the training batches are too large to
        materialise all at once.

        `produce(i)` writes batch i to a CSV and returns its path; the batch is
        fed to one online update and then deleted, so peak disk is one day
        rather than the whole corpus. A full-visibility ATC day is ~1 GB of
        CSV, so holding all 84 would need ~90 GB."""
        from mod_online_update import OnlineUpdateMoD  # online-mod, unmodified
        driver = OnlineUpdateMoD(
            config_file=self.config_path,
            current_cliff=None,
            output_cliff_folder=self.cliff_folder,
            save_fig_folder=self.fig_folder,
        )
        last = None
        for i in range(n_batches):
            csv_path = produce(i)
            batch = _load_online_csv_as_batch(csv_path)
            tag = f"batch_{i:04d}"
            if verbose:
                print(f"[online-cliff] batch {i+1}/{n_batches}: {len(batch)} dets "
                      f"from {pathlib.Path(csv_path).name}", flush=True)
            driver.updateMoD(batch, tag)
            last = os.path.join(self.cliff_folder, f"{tag}.csv")
            if cleanup_batches:
                try:
                    os.unlink(csv_path)
                except OSError:
                    pass
        self._final_snapshot = last
        return last

    def cell_key(self, x, y):
        """Snap a query point to its nearest grid-cell centre (the online map's
        cells are centred on the k*step lattice). Rounded to 3 decimals to match
        the centres the builder stores."""
        s = self.step
        cx = round(round(x / s) * s, 3)
        cy = round(round(y / s) * s, 3)
        return (cx, cy)

    def _read_snapshot(self):
        """final snapshot CSV -> {cell_key: [Comp, ...]}, the full 2-D SWGMM.

        Column layout written by the online updater (utils.read_cliff_map_data):
        x, y, mean_speed, mean_heading, cov_ss, cov_sh, cov_hs, cov_hh, weight,
        motion_ratio. The online updater orders the mixture dimensions
        (speed, heading); cliffmap.cell.CliffMapCell orders them
        (heading, speed). Comp stores them by name so the two conventions
        cannot be transposed by accident."""
        cells = {}
        if not self._final_snapshot or not os.path.exists(self._final_snapshot):
            return cells
        arr = pd.read_csv(self._final_snapshot, header=None).to_numpy()
        for row in arr:
            key = (round(float(row[0]), 3), round(float(row[1]), 3))
            cells.setdefault(key, []).append(Comp(
                w=float(row[8]),
                mean_h=float(row[3]), var_h=float(row[7]),
                mean_s=float(row[2]), var_s=float(row[4]),
                cov_sh=float(row[5])))
        return cells

    def _heading_density(self, comps):
        """720-point heading marginal of a cell's SWGMM: sum over components of
        w * wrapped-normal(mean_heading, var_heading), winding -W..W. Mirrors
        cliffmap.cell.CliffMapCell.heading_probability."""
        W = int(self.params["wind_num"])
        ks = np.arange(-W, W + 1)
        dens = np.zeros(G_GRID)
        for c in comps:
            if c.var_h <= 0:
                continue
            # d[g, k] = theta_g + 2*pi*k - mean_h ; wrapped normal summed over windings
            d = THETA_G[:, None] + 2.0 * math.pi * ks[None, :] - c.mean_h
            dens += c.w * np.exp(-0.5 * d * d / c.var_h).sum(axis=1) / math.sqrt(2.0 * math.pi * c.var_h)
        return dens

    def _cell_densities(self, comps, theta, rho):
        """(p_heading, p_speed, p_joint, mean_rho) for one cell over arrays of
        observed (theta, rho). Mirrors cliffmap.cell.CliffMapCell's
        heading_probability / speed_probability / probability: the heading
        marginal is a wrapped normal on (mean_h, var_h), the speed marginal an
        ordinary normal on (mean_s, var_s), and the joint the semi-wrapped
        bivariate normal summed over the same windings."""
        W = int(self.params["wind_num"])
        ks = np.arange(-W, W + 1)
        theta = np.asarray(theta, float)
        rho = np.asarray(rho, float)
        p_h = np.zeros_like(theta)
        p_s = np.zeros_like(theta)
        p_j = np.zeros_like(theta)
        mean_rho = 0.0
        for c in comps:
            mean_rho += c.w * c.mean_s
            if c.var_h > 0:
                d = theta[:, None] + 2.0 * math.pi * ks[None, :] - c.mean_h
                p_h += c.w * np.exp(-0.5 * d * d / c.var_h).sum(axis=1) / math.sqrt(2.0 * math.pi * c.var_h)
            if c.var_s > 0:
                ds = rho - c.mean_s
                p_s += c.w * np.exp(-0.5 * ds * ds / c.var_s) / math.sqrt(2.0 * math.pi * c.var_s)
            det = c.var_h * c.var_s - c.cov_sh * c.cov_sh
            if det > 0:
                dh = theta[:, None] + 2.0 * math.pi * ks[None, :] - c.mean_h
                ds = (rho - c.mean_s)[:, None]
                # quadratic form of the 2x2 inverse covariance, (heading, speed)
                q = (c.var_s * dh * dh - 2.0 * c.cov_sh * dh * ds + c.var_h * ds * ds) / det
                p_j += c.w * np.exp(-0.5 * q).sum(axis=1) / (2.0 * math.pi * math.sqrt(det))
        return p_h, p_s, p_j, mean_rho

    def build_scorer(self, n_grid=G_GRID):
        """Precompute the GridHeadingScorer over every fitted cell. Cells whose
        reconstructed density is non-positive are dropped (treated as unmatched,
        the same coverage policy as the batch baseline)."""
        gs = GridHeadingScorer(n_grid=n_grid)
        cells = self._read_snapshot()
        n_ok = 0
        for key, comps in cells.items():
            dens = self._heading_density(comps)
            if dens.sum() <= 0:
                continue
            gs.add_distribution(key, dens)
            n_ok += 1
        self._scorer = gs
        self.n_cells = len(cells)
        self.n_cells_scored = n_ok
        return gs

    def build_flow_model(self):
        """Keep the fitted mixture per cell for the log-density channels. Cells
        are dropped on the same rule build_scorer uses, so `matched` means the
        same thing for every channel and for CRPS."""
        self._flow_cells = {}
        for key, comps in self._read_snapshot().items():
            if self._heading_density(comps).sum() > 0:
                self._flow_cells[key] = comps
        return self._flow_cells

    def score_flow(self, x, y, theta, rho, gs=None):
        """Score a chunk on every channel the main flow table reports.

        Returns a dict of per-detection arrays: matched, crps, mpp,
        log_p_heading, log_p_speed, log_p_joint, mean_rho_pred. Unmatched rows
        carry the uniform CRPS/MPP floors and placeholder 0.0 log-densities plus
        NaN speed, which is the convention FlowAccumulator expects: it
        overwrites them with the shared coverage floor."""
        gs = gs or self._scorer
        cells = getattr(self, "_flow_cells", None)
        if cells is None:
            cells = self.build_flow_model()
        theta = np.asarray(theta, float)
        rho = np.asarray(rho, float)
        keys = [self.cell_key(float(xi), float(yi)) for xi, yi in zip(x, y)]
        matched = np.array([k in gs._dists for k in keys], dtype=bool)
        crps, mpp = gs.score(keys, matched, theta)

        n = len(keys)
        lp_h = np.zeros(n); lp_s = np.zeros(n); lp_j = np.zeros(n)
        mean_rho = np.full(n, np.nan)
        # group the chunk by cell so each mixture is evaluated once
        by_cell = collections.defaultdict(list)
        for i, (k, m) in enumerate(zip(keys, matched)):
            if m:
                by_cell[k].append(i)
        for key, idx in by_cell.items():
            comps = cells.get(key)
            if not comps:
                continue
            idx = np.asarray(idx)
            p_h, p_s, p_j, m_rho = self._cell_densities(comps, theta[idx], rho[idx])
            with np.errstate(divide="ignore"):
                lp_h[idx] = np.where(p_h > 0, np.log(np.maximum(p_h, 1e-300)), LOG_FLOOR)
                lp_s[idx] = np.where(p_s > 0, np.log(np.maximum(p_s, 1e-300)), LOG_FLOOR)
                lp_j[idx] = np.where(p_j > 0, np.log(np.maximum(p_j, 1e-300)), LOG_FLOOR)
            mean_rho[idx] = m_rho
        return {"matched": matched, "crps": np.asarray(crps), "mpp": np.asarray(mpp),
                "log_p_heading": lp_h, "log_p_speed": lp_s, "log_p_joint": lp_j,
                "mean_rho_pred": mean_rho}

    def score_detections(self, x, y, theta, gs=None):
        """Score a chunk of test detections: map each to its cell, mark matched
        iff that cell was fitted, and return (crps, mpp) via GridHeadingScorer
        (uniform floor pi/4, 1/8 on unmatched)."""
        gs = gs or self._scorer
        keys = [self.cell_key(float(xi), float(yi)) for xi, yi in zip(x, y)]
        matched = [k in gs._dists for k in keys]
        crps, mpp = gs.score(keys, matched, theta)
        return np.asarray(crps), np.asarray(mpp), np.asarray(matched, bool)

    def cleanup(self):
        if self._own_work and os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)
