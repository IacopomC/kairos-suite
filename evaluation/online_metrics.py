"""Incremental (streaming) metric accumulators for the online evaluator.

Each accumulator is fed per-detection score arrays chunk by chunk and holds only
running sums / counts / threshold-tallies, so a full-length test set is scored
without materialising a per-detection predictions file. These are the single
source of truth for every metric the evaluator reports (flow MLPD + speed MAE,
speed calibration, coverage-stratified MLPD, edge-coupling, presence).

The floors for the coverage-aware MLPD use the max observed speed over the test
set, clipped to a physical pedestrian maximum (DEFAULT_SPEED_CAP_MPS) so a single
tracker glitch cannot set the floor. Because every method scores the identical
held-out detections with the same cap, each run's running rho_max is the same, so
the floor is identical across methods (matching the pooled rho_max the batch path
used).

Matched log-densities are bounded below by a shared per-detection density floor
(DEFAULT_DENSITY_FLOOR), so the worst-case charge for a single detection is the
same for every method regardless of the numerical guard each model applies
internally. The fraction of matched detections falling below the floor is
reported per channel alongside the MLPD.
"""
import hashlib
import math
from collections import deque

import numpy as np
from scipy.special import betaln, erf, gammaln

_CH = ("heading", "speed", "joint")
_SQRT2 = math.sqrt(2.0)
_PRESENCE_EPS = 1e-9

# Physical pedestrian max speed (m/s). The tracker occasionally emits non-physical
# outliers (~10 m/s); left unclipped, a single glitch sets rho_max and therefore
# the shared coverage-aware MLPD floor for EVERY method. FlowAccumulator clips
# observed speeds to this before rho_max / speed-MAE so floors are physical and
# identical across methods. Override per accumulator; 0/None disables.
DEFAULT_SPEED_CAP_MPS = 3.0

# Shared per-detection density floor for matched log-densities, applied
# identically to every channel and every method inside FlowAccumulator. Below
# a density this small the prediction already says "essentially impossible";
# how far below carries no further information about forecast quality, and
# without a shared floor the worst-case per-detection charge is set by each
# model's internal numerical guard rather than by the benchmark. The value
# matches the per-bin probability floor of the histogram baselines and the
# presence-channel epsilon (_PRESENCE_EPS). Override per accumulator; 0/None
# disables (raw densities).
DEFAULT_DENSITY_FLOOR = 1e-9


def clip_speed(rho, cap):
    """Cap observed speeds at a physical pedestrian maximum. NaN/inf (unmatched or
    missing speeds) are preserved for downstream finite-masking; cap<=0/None is a
    no-op."""
    if cap is None or cap <= 0:
        return rho
    return np.minimum(np.asarray(rho, dtype=float), float(cap))


def htag(horizon_s):
    """Canonical string tag for an event horizon in seconds: '5', '10', '60',
    '0.5'. Matches the p<tag> key naming of the presence-grid binding."""
    return f"{float(horizon_s):g}"


class PresenceAccumulator:
    """Streaming presence/occupancy calibration: Bernoulli log-loss plus the
    reliability/resolution (Murphy) decomposition, for each named presence
    forecast at the configured event horizons.

    Bernoulli-LL is an exact running sum. The decomposition uses FIXED
    probability bins (linspace 0..1) so it streams in one pass; the binning is a
    reporting choice and the reliability/resolution identities are unchanged. The
    base-rate (constant) estimator is a closed form of the outcome mean and the
    supplied training rate, so it needs no per-bin tally."""

    def __init__(self, n_bins: int = 20, base_rate_5s=None, base_rate_10s=None,
                 estimators=("kairos",), horizons=(5.0, 10.0),
                 base_rates=None):
        self.n_bins = n_bins
        self.edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.horizons = tuple(float(h) for h in horizons)
        self.tags = [htag(h) for h in self.horizons]
        # Per-horizon base rate, keyed by tag. `base_rates` (a dict keyed by
        # horizon seconds) wins; the 5 s / 10 s kwargs fill their slots.
        self.base_rate = {t: None for t in self.tags}
        for t in self.tags:
            if base_rates is not None:
                for k, v in base_rates.items():
                    if htag(k) == t:
                        self.base_rate[t] = v
                        break
            if self.base_rate[t] is None:
                self.base_rate[t] = {"5": base_rate_5s, "10": base_rate_10s}.get(t)
        self.est_names = list(estimators)
        # per horizon: n, sum_y, and an est dict per named estimator.
        self._h = {}
        for tag in self.tags:
            self._h[tag] = {"n": 0, "sum_y": 0.0}
            for nm in self.est_names:
                self._h[tag][nm] = self._new_est(n_bins)

    @staticmethod
    def _new_est(n_bins):
        return {"bll": 0.0, "hbll": np.zeros(24),
                "bin_p": np.zeros(n_bins), "bin_y": np.zeros(n_bins),
                "bin_n": np.zeros(n_bins, dtype=np.int64)}

    def _add_est(self, est, p, y, hour=None):
        p = np.clip(p, _PRESENCE_EPS, 1 - _PRESENCE_EPS)
        b_ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        est["bll"] += b_ll
        if hour is not None:
            est["hbll"][hour] += b_ll
        b = np.clip(np.digitize(p, self.edges[1:-1], right=False), 0, self.n_bins - 1)
        np.add.at(est["bin_p"], b, p)
        np.add.at(est["bin_y"], b, y)
        np.add.at(est["bin_n"], b, 1)

    def update(self, y_by_tag, preds, hour=None):
        """y_by_tag: {tag: outcome array} per horizon; preds: {name: {tag:
        probability array}} for each estimator. `hour` (int 0-23, the
        query's local hour of day) additionally folds this batch into the
        per-hour tallies; omit it and the result is unchanged."""
        for tag in self.tags:
            y = np.asarray(y_by_tag[tag], dtype=float)
            if y.size == 0:
                continue
            h = self._h[tag]
            h["n"] += int(y.size); h["sum_y"] += float(y.sum())
            if hour is not None:
                h.setdefault("h_n", np.zeros(24, dtype=np.int64))
                h.setdefault("h_y", np.zeros(24))
                h["h_n"][hour] += int(y.size)
                h["h_y"][hour] += float(y.sum())
            for nm in self.est_names:
                self._add_est(h[nm], np.asarray(preds[nm][tag], dtype=float), y,
                              hour=hour)

    def _decompose(self, est, n, obar):
        nb = est["bin_n"]
        m = nb > 0
        pk = est["bin_p"][m] / nb[m]
        ok = est["bin_y"][m] / nb[m]
        nk = nb[m]
        rel = float(np.sum(nk * (pk - ok) ** 2) / n)
        res = float(np.sum(nk * (ok - obar) ** 2) / n)
        return rel, res

    def result(self):
        out = {}
        for tag in self.tags:
            h = self._h[tag]
            n = h["n"]
            if n == 0:
                out[tag] = None
                continue
            obar = h["sum_y"] / n
            unc = obar * (1.0 - obar)
            res_tag = {"occupancy_rate": obar, "uncertainty": unc, "n": n}
            for nm in self.est_names:
                est = h[nm]
                rel, res = self._decompose(est, n, obar)
                res_tag[nm] = {"bern_ll": est["bll"] / n,
                               "reliability": rel, "resolution": res}
            # Base-rate constant c: closed form from obar (+ supplied training rate).
            c = self.base_rate[tag]
            c_insample = c is None
            if c is None:
                c = obar
            cc = min(max(c, _PRESENCE_EPS), 1 - _PRESENCE_EPS)
            res_tag["base_rate"] = {
                "bern_ll": obar * math.log(cc) + (1 - obar) * math.log(1 - cc),
                "reliability": (cc - obar) ** 2, "resolution": 0.0,
                "constant": c, "in_sample": c_insample}
            # Per-hour fold, present only when update() was fed hours.
            if "h_n" in h and h["h_n"].sum() > 0:
                hn = h["h_n"]
                hourly = {"n": hn.tolist(),
                          "occ": [float(h["h_y"][k] / hn[k]) if hn[k] else None
                                  for k in range(24)]}
                for nm in self.est_names:
                    hourly[nm] = [float(self._h[tag][nm]["hbll"][k] / hn[k])
                                  if hn[k] else None for k in range(24)]
                hourly["base_rate"] = [
                    (h["h_y"][k] / hn[k]) * math.log(cc)
                    + (1 - h["h_y"][k] / hn[k]) * math.log(1 - cc)
                    if hn[k] else None for k in range(24)]
                res_tag["hourly"] = hourly
            out[tag] = res_tag
        return out


class HeadingDistributionAccumulator:
    """Streaming circular CRPS and MPP for the fixed-center heading mixture.

    Circular CRPS (Grimit et al., 2006): with arc distance d(a, b) in [0, pi],
        CRPS(F, y) = E_F d(X, y) - 0.5 * E_F d(X, X')
    -- the expected arc distance of the forecast mass from the observation,
    minus a spread term that makes the score proper. Lower is better; bounded
    by pi; distance-aware (mass 30 deg off costs less than mass 180 deg off),
    unlike the local log score.

    MPP (Mean Predictive Probability, Verdoja et al.): the probability mass the
    forecast places on the observed 45-degree sector (uniform = 1/8). Higher is
    better; comparable across histogram and continuous methods; NOT proper (it
    rewards concentrating mass on the modal bin), so it is a comparability
    secondary, never the primary.

    The forecast is the model's K=8 slot mixture with fixed centers k*pi/4 and
    the circular-distance Gaussian kernel (sigma_theta) used everywhere else in
    this module; both scores reduce to table lookups per detection. Unmatched
    detections receive the uniform-forecast values (CRPS pi/4, MPP 1/8), the
    same coverage symmetry as the MLPD floor.
    """

    _K = 8

    def __init__(self, sigma_theta=0.4, n_grid=720):
        assert n_grid % (2 * self._K) == 0
        self.G = n_grid
        g = np.arange(n_grid)
        theta_g = g * (2.0 * math.pi / n_grid)
        d0 = np.minimum(theta_g, 2.0 * math.pi - theta_g)      # arc dist to 0
        f0 = np.exp(-0.5 * d0 ** 2 / sigma_theta ** 2)
        f0 /= f0.sum()                                          # mass per grid cell
        arc = np.minimum.outer(theta_g, np.zeros(1)).squeeze()  # placeholder
        # arc distance between grid points i and j depends on (i - j) mod G:
        arc_off = np.minimum(theta_g, 2.0 * math.pi - theta_g)  # d(offset)
        # T[j] = E_{X ~ f0} d(X, theta_j): expected arc distance from a
        # component centered at 0 to a fixed angle theta_j.
        idx = (np.arange(n_grid)[:, None] - np.arange(n_grid)[None, :]) % n_grid
        self.T = (f0[None, :] * arc_off[idx]).sum(axis=1)       # (G,)
        # D2[k, l] = E d(X_k, X_l) for components centered k*pi/4 and l*pi/4:
        # E arc(A - B + mu_k - mu_l), A, B ~ f0. Depends only on (k - l) mod K.
        step = n_grid // self._K
        conv = np.zeros(n_grid)                                 # density of A - B
        for s in range(n_grid):
            conv[s] = float((f0 * f0[(np.arange(n_grid) - s) % n_grid]).sum())
        d2_off = np.array([(conv * arc_off[(np.arange(n_grid) + k * step) % n_grid]).sum()
                           for k in range(self._K)])
        self.D2 = d2_off[(np.arange(self._K)[:, None] - np.arange(self._K)[None, :]) % self._K]
        # M8[k, b] = mass of component k inside sector b (sector b = angles
        # within pi/8 of b*pi/4, the nearest-slot assignment).
        sector = ((g + step // 2) // step) % self._K            # sector of each grid pt
        self.M8 = np.zeros((self._K, self._K))
        for k in range(self._K):
            fk = f0[(g - k * step) % n_grid]
            for b in range(self._K):
                self.M8[k, b] = fk[sector == b].sum()
        self.step = step
        self.crps_unif = math.pi / 4.0
        self.mpp_unif = 1.0 / self._K
        self._sum = {"crps": 0.0, "mpp": 0.0, "crps_m": 0.0, "mpp_m": 0.0}
        self._n = 0
        self._n_m = 0

    def rows(self, matched, theta_obs, slot_weights):
        """Per-detection (crps, mpp, good, W): the scores update() accumulates,
        plus the dense normalized weight matrix (zero rows where not good) for
        callers that derive further per-row quantities (sector masses)."""
        theta = np.mod(np.asarray(theta_obs, float), 2.0 * math.pi)
        matched = np.asarray(matched, bool)
        n = theta.size
        # Ragged slot weights -> dense (n, K); rows without K weights count as
        # unmatched (uniform floor), matching the coverage policy.
        W = np.zeros((n, self._K))
        ok = matched.copy()
        for i in range(n):
            row = slot_weights[i] if slot_weights is not None else []
            if ok[i] and len(row) == self._K:
                W[i] = row
            else:
                ok[i] = False
        s = W.sum(axis=1)
        good = ok & (s > 0) & np.isfinite(theta)
        W[good] /= s[good, None]
        W[~good] = 0.0
        gi = np.round(theta / (2.0 * math.pi / self.G)).astype(np.int64) % self.G
        centers = (np.arange(self._K) * self.step)[None, :]
        t_lookup = self.T[(gi[:, None] - centers) % self.G]     # (n, K)
        term1 = (W * t_lookup).sum(axis=1)
        term2 = 0.5 * np.einsum("nk,kl,nl->n", W, self.D2, W)
        crps = np.where(good, term1 - term2, self.crps_unif)
        b_obs = np.round(theta / (math.pi / 4.0)).astype(np.int64) % self._K
        mpp = np.where(good, (W * self.M8[:, b_obs].T).sum(axis=1), self.mpp_unif)
        bad = ~np.isfinite(theta)
        crps = np.where(bad, self.crps_unif, crps)
        mpp = np.where(bad, self.mpp_unif, mpp)
        return crps, mpp, good, W

    def sector_masses(self, W):
        """Mixture mass on each of the K shared centred sectors, per row:
        the binned form of the same forecast (rows of zeros stay zeros)."""
        return W @ self.M8

    def accumulate(self, crps, mpp, good):
        self._sum["crps"] += float(crps.sum()); self._sum["mpp"] += float(mpp.sum())
        self._sum["crps_m"] += float(crps[good].sum()); self._sum["mpp_m"] += float(mpp[good].sum())
        self._n += int(crps.size); self._n_m += int(good.sum())

    def update(self, matched, theta_obs, slot_weights):
        crps, mpp, good, _ = self.rows(matched, theta_obs, slot_weights)
        self.accumulate(crps, mpp, good)

    def result(self):
        if self._n == 0:
            return None
        r = {"n": self._n, "n_matched": self._n_m,
             "crps_rad": self._sum["crps"] / self._n,
             "mpp": self._sum["mpp"] / self._n,
             "crps_uniform": self.crps_unif, "mpp_uniform": self.mpp_unif}
        if self._n_m:
            r["crps_rad_matched_only"] = self._sum["crps_m"] / self._n_m
            r["mpp_matched_only"] = self._sum["mpp_m"] / self._n_m
        return r


class HistogramHeadingAccumulator:
    """Streaming circular CRPS + MPP for EDGE-ALIGNED histogram heading
    forecasts: bin b covers [b*2pi/K, (b+1)*2pi/K). Feed the
    per-detection bin masses in that standard indexing; both scores reduce to
    table lookups. MPP uses the SHARED sector partition of the paper (sectors
    centred on k*2pi/K, the same partition every method is scored on): for an
    edge-aligned histogram the mass on centred sector s is
    (m_{s-1} + m_s) / 2 under the histogram's own uniform-within-bin density.
    Unmatched detections receive the uniform floors (CRPS pi/4, MPP 1/K)."""

    def __init__(self, n_bins=8, n_grid=720):
        assert n_grid % (2 * n_bins) == 0
        self.K = n_bins
        self.G = n_grid
        g = np.arange(n_grid)
        theta_g = g * (2.0 * math.pi / n_grid)
        arc_off = np.minimum(theta_g, 2.0 * math.pi - theta_g)
        step = n_grid // n_bins
        # Component b = uniform on grid points [b*step, (b+1)*step).
        # T[b, j] = E_{X ~ unif bin b} arc(X, theta_j); D[b, c] = E arc(X_b, X_c).
        self.T = np.empty((n_bins, n_grid))
        for b in range(n_bins):
            pts = np.arange(b * step, (b + 1) * step)
            self.T[b] = arc_off[(pts[:, None] - np.arange(n_grid)[None, :]) % n_grid].mean(axis=0)
        self.D = np.empty((n_bins, n_bins))
        for b in range(n_bins):
            pts_b = np.arange(b * step, (b + 1) * step)
            for c in range(n_bins):
                self.D[b, c] = self.T[c][pts_b].mean()
        self._sum = {"crps": 0.0, "mpp": 0.0, "crps_m": 0.0, "mpp_m": 0.0}
        self._n = 0
        self._n_m = 0

    def rows(self, matched, theta_obs, masses):
        """Per-detection (crps, mpp, good): the scores update() accumulates."""
        theta = np.mod(np.asarray(theta_obs, float), 2.0 * math.pi)
        matched = np.asarray(matched, bool)
        m = np.asarray(masses, float)
        n = theta.size
        s = m.sum(axis=1)
        good = matched & (s > 0) & np.isfinite(theta)
        mm = np.zeros_like(m)
        mm[good] = m[good] / s[good, None]
        gi = np.round(theta / (2.0 * math.pi / self.G)).astype(np.int64) % self.G
        term1 = np.einsum("nk,kn->n", mm, self.T[:, gi])
        term2 = 0.5 * np.einsum("nk,kl,nl->n", mm, self.D, mm)
        crps = np.where(good, term1 - term2, math.pi / 4.0)
        # Shared centred sector s_obs; histogram mass on it = (m_{s-1}+m_s)/2.
        s_obs = np.round(theta / (2.0 * math.pi / self.K)).astype(np.int64) % self.K
        mpp_v = 0.5 * (mm[np.arange(n), (s_obs - 1) % self.K] + mm[np.arange(n), s_obs])
        mpp = np.where(good, mpp_v, 1.0 / self.K)
        return crps, mpp, good

    def accumulate(self, crps, mpp, good):
        self._sum["crps"] += float(crps.sum()); self._sum["mpp"] += float(mpp.sum())
        self._sum["crps_m"] += float(crps[good].sum()); self._sum["mpp_m"] += float(mpp[good].sum())
        self._n += int(crps.size); self._n_m += int(good.sum())

    def update(self, matched, theta_obs, masses):
        crps, mpp, good = self.rows(matched, theta_obs, masses)
        self.accumulate(crps, mpp, good)

    def result(self):
        if self._n == 0:
            return None
        r = {"n": self._n, "n_matched": self._n_m,
             "crps_rad": self._sum["crps"] / self._n,
             "mpp": self._sum["mpp"] / self._n,
             "crps_uniform": math.pi / 4.0, "mpp_uniform": 1.0 / self.K}
        if self._n_m:
            r["crps_rad_matched_only"] = self._sum["crps_m"] / self._n_m
            r["mpp_matched_only"] = self._sum["mpp_m"] / self._n_m
        return r


class BinnedControlAccumulator:
    """CRPS + MPP of the BINNED form of a mixture forecast: the mixture's mass
    on each shared centred sector, spread uniformly within the sector. Scored
    next to the continuous mixture on the same rows, it separates where a
    heading loss comes from: if the binned and continuous scores agree, the
    sector-level weights carry the result; if the binned score is better, the
    within-sector kernel shape is the cost.

    A centred-sector histogram is an edge-aligned histogram in the rotated
    coordinate theta' = theta + pi/K (arc distances are rotation-invariant),
    so the CRPS reuses HistogramHeadingAccumulator's tables under that
    rotation. The MPP of the binned forecast on the observed centred sector
    is the sector's own mass. Uniform floors as everywhere (pi/4, 1/K)."""

    def __init__(self, n_bins=8, n_grid=720):
        self.K = n_bins
        self._hist = HistogramHeadingAccumulator(n_bins=n_bins, n_grid=n_grid)
        self._sum = {"crps": 0.0, "mpp": 0.0, "crps_m": 0.0, "mpp_m": 0.0}
        self._n = 0
        self._n_m = 0

    def update(self, matched, theta_obs, sector_masses, good):
        theta = np.asarray(theta_obs, float)
        rot = theta + math.pi / self.K
        crps, _, good_h = self._hist.rows(good, rot, sector_masses)
        # good_h == good except where sector masses sum to 0 (never for a
        # normalized mixture row); keep the caller's mask for MPP symmetry.
        s_obs = np.round(np.mod(theta, 2.0 * math.pi) / (2.0 * math.pi / self.K)).astype(np.int64) % self.K
        m = np.asarray(sector_masses, float)
        mpp = np.where(good_h, m[np.arange(theta.size), s_obs], 1.0 / self.K)
        self._sum["crps"] += float(crps.sum()); self._sum["mpp"] += float(mpp.sum())
        self._sum["crps_m"] += float(crps[good_h].sum()); self._sum["mpp_m"] += float(mpp[good_h].sum())
        self._n += int(theta.size); self._n_m += int(good_h.sum())

    def result(self):
        if self._n == 0:
            return None
        r = {"n": self._n, "n_matched": self._n_m,
             "crps_rad": self._sum["crps"] / self._n,
             "mpp": self._sum["mpp"] / self._n,
             "crps_uniform": math.pi / 4.0, "mpp_uniform": 1.0 / self.K}
        if self._n_m:
            r["crps_rad_matched_only"] = self._sum["crps_m"] / self._n_m
            r["mpp_matched_only"] = self._sum["mpp_m"] / self._n_m
        return r


class PassageWeightedAccumulator:
    """Per-passage reweighting of per-detection heading scores: one vote per
    (track id, voxel) crossing instead of one per frame.

    A passage is a maximal run of consecutive detections of one track id
    inside one voxel; a new passage starts when the id enters a different
    voxel or after a time gap larger than `gap_s` (which also separates track
    ids reused across recording days). Each passage contributes the MEAN of
    its rows' scores once (uniform floors included, the same coverage policy
    as the per-frame column), and the aggregate is the mean over passages.
    The per-frame mean weights a crossing by its frame count, so dwellers and
    slow walkers dominate it; this column weights every crossing equally --
    the frames-are-not-passages correction of the measurement study applied
    to scoring. Matched-only means are over each passage's matched rows
    (passages with no matched row enter only the floor-inclusive column)."""

    def __init__(self, voxel_size=0.4, gap_s=60.0):
        self.vs = voxel_size
        self.gap = gap_s
        self._open = {}   # id -> [vx, vy, t_last, n, sum3, n_m, sum3_m]
        self._n_pass = 0
        self._sum = np.zeros(3)      # passage means of (lp, crps, mpp), floors included
        self._n_pass_m = 0
        self._sum_m = np.zeros(3)

    def _close(self, st):
        if st[3]:
            self._n_pass += 1
            self._sum += st[4] / st[3]
        if st[5]:
            self._n_pass_m += 1
            self._sum_m += st[6] / st[5]

    def update(self, t_s, xyz, ids, matched, lp_full, crps, mpp):
        """lp_full: heading log-density with the uniform floor (-log 2pi)
        already substituted on unmatched rows, so passage means share the
        coverage policy of every other column."""
        t = np.asarray(t_s, float)
        xy = np.asarray(xyz, float)
        vx = np.floor(xy[:, 0] / self.vs).astype(np.int64)
        vy = np.floor(xy[:, 1] / self.vs).astype(np.int64)
        matched = np.asarray(matched, bool)
        vals = np.stack([np.asarray(lp_full, float),
                         np.asarray(crps, float),
                         np.asarray(mpp, float)], axis=1)
        for i in range(t.size):
            pid = ids[i] if ids is not None else -1
            st = self._open.get(pid)
            if st is not None and (st[0] != vx[i] or st[1] != vy[i]
                                   or t[i] - st[2] > self.gap):
                self._close(st)
                st = None
            if st is None:
                st = [vx[i], vy[i], t[i], 0, np.zeros(3), 0, np.zeros(3)]
                self._open[pid] = st
            st[2] = t[i]
            st[3] += 1
            st[4] += vals[i]
            if matched[i]:
                st[5] += 1
                st[6] += vals[i]

    def result(self):
        for st in self._open.values():
            self._close(st)
        self._open.clear()
        if self._n_pass == 0:
            return None
        r = {"n_passages": self._n_pass,
             "mlpd_heading": self._sum[0] / self._n_pass,
             "crps_rad": self._sum[1] / self._n_pass,
             "mpp": self._sum[2] / self._n_pass}
        if self._n_pass_m:
            r["n_passages_matched"] = self._n_pass_m
            r["mlpd_heading_matched_only"] = self._sum_m[0] / self._n_pass_m
            r["crps_rad_matched_only"] = self._sum_m[1] / self._n_pass_m
            r["mpp_matched_only"] = self._sum_m[2] / self._n_pass_m
        return r


class CellMeanAccumulator:
    """Per-cell equal-weight aggregation of the per-detection heading scores:
    the aggregation of the exploration evaluation of Molina et al. (T-RO), where
    every occupied cell contributes its own mean score once regardless of how
    many detections it carries. A detection-weighted mean is dominated by the
    few busy cells that produce almost all detections, so a policy that improves
    rarely-observed cells is invisible in it; weighting every cell equally gives
    those cells a vote, which is what an exploration policy is meant to affect.

    Scores are taken against the actual held-out detections (the ground truth is
    each real heading, not a reference model, and is not quantized), with the
    same uniform floor on unmatched rows as every other column, so a cell the
    map never learned is penalized rather than skipped. result() returns the
    equal-weight means, a bootstrap CI over cells (statistical power from one
    run), the means restricted to a minimum per-cell detection count, and the
    per-cell CRPS keyed to voxel coordinates for a spatial heatmap."""

    def __init__(self, voxel_size=0.4):
        self.vs = voxel_size
        # voxel key (vx, vy) -> [n, sum_lp, sum_crps, sum_mpp, n_matched]
        self._cells = {}

    def update(self, xyz, good, lp_full, crps, mpp):
        """lp_full carries the uniform heading floor on unmatched rows already,
        matching the coverage policy of the per-detection columns."""
        xy = np.asarray(xyz, float)
        vx = np.floor(xy[:, 0] / self.vs).astype(np.int64)
        vy = np.floor(xy[:, 1] / self.vs).astype(np.int64)
        good = np.asarray(good, bool)
        lp = np.asarray(lp_full, float)
        cr = np.asarray(crps, float)
        mp = np.asarray(mpp, float)
        for i in range(vx.size):
            key = (int(vx[i]), int(vy[i]))
            st = self._cells.get(key)
            if st is None:
                st = [0, 0.0, 0.0, 0.0, 0]
                self._cells[key] = st
            st[0] += 1
            st[1] += lp[i]
            st[2] += cr[i]
            st[3] += mp[i]
            if good[i]:
                st[4] += 1

    def result(self, boot_n=2000, boot_seed=0, thresholds=(0, 5, 25, 100)):
        if not self._cells:
            return None
        keys = np.array(list(self._cells.keys()), dtype=np.int64)      # (C, 2)
        vals = np.array(list(self._cells.values()), dtype=float)       # (C, 5)
        n = vals[:, 0]
        lp = vals[:, 1] / n
        crps = vals[:, 2] / n
        mpp = vals[:, 3] / n
        rng = np.random.default_rng(boot_seed)
        C = len(n)
        boot = np.array([crps[rng.integers(0, C, C)].mean() for _ in range(boot_n)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        by_thresh = {}
        for t in thresholds:
            keep = n >= t
            if keep.any():
                by_thresh[int(t)] = {"n_cells": int(keep.sum()),
                                     "crps_rad": float(crps[keep].mean()),
                                     "mlpd_heading": float(lp[keep].mean()),
                                     "mpp": float(mpp[keep].mean())}
        return {"n_cells": C,
                "mlpd_heading": float(lp.mean()),
                "crps_rad": float(crps.mean()),
                "mpp": float(mpp.mean()),
                "crps_ci95": (float(lo), float(hi)),
                "median_det_per_cell": float(np.median(n)),
                "by_threshold": by_thresh,
                # per-cell arrays for the spatial heatmap (voxel_size scale):
                "cell_vx": keys[:, 0], "cell_vy": keys[:, 1],
                "per_cell_crps": crps, "per_cell_mlpd": lp,
                "per_cell_mpp": mpp, "per_cell_n": n}


class GridHeadingScorer:
    """Circular CRPS + MPP for an ARBITRARY heading density given on a regular
    angular grid -- the baseline-side counterpart of
    HeadingDistributionAccumulator (which is specialised to the kairos slot
    mixture). A baseline dumps each distinct predictive distribution once
    (per cell, or per cell and time bin for spectral methods) as `n_grid`
    probability masses; `add_distribution` precomputes its expected-distance
    profile, and `score` is then a table lookup per detection, so scoring cost
    does not depend on how many detections share a distribution.

    For an 8-bin histogram the grid density is the bin mass
    spread uniformly over each 45-degree sector; for a GMM forecast
    it is the normalised density evaluated on the grid. Same uniform floors
    as the kairos path (CRPS pi/4, MPP 1/8 for unmatched detections)."""

    def __init__(self, n_grid=720, k_bins=8):
        self.G = n_grid
        self.K = k_bins
        g = np.arange(n_grid)
        theta_g = g * (2.0 * math.pi / n_grid)
        self._arc = np.minimum(theta_g, 2.0 * math.pi - theta_g)   # d(offset)
        self._idx = (np.arange(n_grid)[:, None] - np.arange(n_grid)[None, :]) % n_grid
        step = n_grid // k_bins
        self._sector = ((g + step // 2) // step) % k_bins
        self._dists = {}      # key -> (T_vector, half_spread, bin_mass[K])

    def add_distribution(self, key, p_grid):
        p = np.asarray(p_grid, float)
        p = p / p.sum()
        T = (p[None, :] * self._arc[self._idx]).sum(axis=1)        # E d(X, theta_j)
        spread = float(p @ (self._arc[self._idx] @ p))             # E d(X, X')
        bins = np.array([p[self._sector == b].sum() for b in range(self.K)])
        self._dists[key] = (T, 0.5 * spread, bins)

    def score(self, keys, matched, theta_obs):
        """Vectorised over detections: `keys[i]` names the distribution that
        scored detection i (ignored where not matched). Returns (crps, mpp)."""
        theta = np.mod(np.asarray(theta_obs, float), 2.0 * math.pi)
        n = theta.size
        crps = np.full(n, math.pi / 4.0)
        mpp = np.full(n, 1.0 / self.K)
        gi = np.round(theta / (2.0 * math.pi / self.G)).astype(np.int64) % self.G
        bi = np.round(theta / (2.0 * math.pi / self.K)).astype(np.int64) % self.K
        for i in range(n):
            if not matched[i]:
                continue
            d = self._dists.get(keys[i])
            if d is None:
                continue
            T, half_spread, bins = d
            crps[i] = T[gi[i]] - half_spread
            mpp[i] = bins[bi[i]]
        return crps, mpp


class FremenOccupancy:
    """FreMEn per-cell occupancy reference (Krajnik et al.): the
    presence analogue. Fits, per cell and horizon, the DC occupancy rate plus a
    spectral coefficient at each candidate period from the training (cell, t,
    occupied) stream, then predicts p(occupied | t) = gamma0 + sum_f 2|c_f|
    cos(omega_f t + angle c_f), clipped to [0,1]. Same spectral primitive as the
    model's NUDFT; the periods are a parameter set per dataset (like the omegas).

    Fit is a single streaming pass (array-backed np.add.at), so it scales to a
    multi-day training replay. Cells unseen in training predict the global rate."""

    def __init__(self, periods_s, horizons=(5.0, 10.0)):
        self.omegas = np.array([2.0 * math.pi / float(P) for P in periods_s])
        self.F = self.omegas.size
        self.horizons = tuple(horizons)
        self._row = {}                       # cell key -> row index
        self._n = {h: [] for h in self.horizons}      # grown lists -> arrays
        self._sy = {h: [] for h in self.horizons}
        self._S = {h: [] for h in self.horizons}      # complex (F,) per row
        self._model = None

    @staticmethod
    def _key(vx, vy, vz):
        return (vx.astype(np.int64) << 42) ^ ((vy.astype(np.int64) & 0x1FFFFF) << 21) \
            ^ (vz.astype(np.int64) & 0x1FFFFF)

    def _rows_for(self, keys):
        rows = np.empty(keys.size, dtype=np.int64)
        for i, k in enumerate(keys.tolist()):
            r = self._row.get(k)
            if r is None:
                r = len(self._row); self._row[k] = r
                for h in self.horizons:
                    self._n[h].append(0.0); self._sy[h].append(0.0)
                    self._S[h].append(np.zeros(self.F, dtype=np.complex128))
            rows[i] = r
        return rows

    def fit_update(self, vx, vy, vz, t, y_by_h):
        """Accumulate one query: cells (vx,vy,vz), time t, occupancy y_by_h[h]."""
        keys = self._key(np.asarray(vx), np.asarray(vy), np.asarray(vz))
        rows = self._rows_for(keys)
        econj = np.exp(-1j * self.omegas * t)          # (F,)
        for h in self.horizons:
            y = np.asarray(y_by_h[h], dtype=float)
            n = np.asarray(self._n[h]); sy = np.asarray(self._sy[h])
            np.add.at(n, rows, 1.0); np.add.at(sy, rows, y)
            self._n[h] = list(n); self._sy[h] = list(sy)
            S = self._S[h]
            for i, r in enumerate(rows.tolist()):
                S[r] += y[i] * econj

    def finalize(self):
        """Freeze per-cell (gamma0, magnitude, phase) arrays for fast prediction."""
        self._model = {}
        for h in self.horizons:
            n = np.asarray(self._n[h]); sy = np.asarray(self._sy[h])
            S = np.asarray(self._S[h]) if self._S[h] else np.zeros((0, self.F), complex)
            nz = np.maximum(n, 1.0)
            g0 = sy / nz
            c = S / nz[:, None]                        # complex coeff per cell,freq
            self._model[h] = {"g0": g0, "mag": np.abs(c), "phase": np.angle(c),
                              "global": float(sy.sum() / max(n.sum(), 1.0))}
        return self

    def predict(self, vx, vy, vz, t, horizon):
        """p(occupied | t) for the given cells at time t (vectorised)."""
        if self._model is None:
            self.finalize()
        md = self._model[horizon]
        keys = self._key(np.asarray(vx), np.asarray(vy), np.asarray(vz))
        rows = np.array([self._row.get(int(k), -1) for k in keys.tolist()], dtype=np.int64)
        seen = rows >= 0
        p = np.full(keys.size, md["global"], dtype=float)
        if seen.any():
            rr = rows[seen]
            ph = md["mag"][rr] * np.cos(self.omegas[None, :] * t + md["phase"][rr])
            p[seen] = md["g0"][rr] + 2.0 * ph.sum(axis=1)
        return np.clip(p, _PRESENCE_EPS, 1.0 - _PRESENCE_EPS)


class FlowAccumulator:
    """Coverage-aware channel MLPD (heading/speed/joint) + coverage + speed MAE.

    Matched log-densities are clamped below at log(density_floor) so every
    method has the same worst-case per-detection charge; the fraction of
    matched detections under the floor is reported per channel
    (below_floor_*), separating "how good is the typical prediction" (MLPD)
    from "how often does the model call the observed motion essentially
    impossible" (below-floor rate)."""

    def __init__(self, speed_cap_mps=DEFAULT_SPEED_CAP_MPS, rho_max_override=0.0,
                 density_floor=DEFAULT_DENSITY_FLOOR):
        self.n_total = 0
        self.n_matched = 0
        self.sum_lp = {c: 0.0 for c in _CH}   # sum of finite matched log-densities
        self.n_fm = {c: 0 for c in _CH}       # count of finite matched, per channel
        self.n_below = {c: 0 for c in _CH}    # matched with density below the floor
        self.speed_ae_sum = 0.0
        self.speed_ae_n = 0
        self.rho_max = 1e-6
        self.speed_cap_mps = speed_cap_mps
        self.density_floor = density_floor
        self._lp_floor = math.log(density_floor) if density_floor else None
        # Fixed floor domain for cross-method comparability: the default rho_max is
        # a running max over the streamed test set, identical across runs ONLY
        # when they stream identical days. An explicit override pins the floor
        # (e.g. to the speed cap) so partial or differently-ordered runs stay
        # comparable; the observed running max is still reported.
        self.rho_max_override = float(rho_max_override)

    def update(self, matched, rho_obs, log_p, mean_rho):
        """`matched` bool array; `log_p` dict channel->array; `mean_rho`,`rho_obs` arrays."""
        m = np.asarray(matched, dtype=bool)
        self.n_total += int(m.size)
        self.n_matched += int(m.sum())
        # Clip observed speed so a tracker glitch cannot set rho_max (the shared
        # MLPD floor) or inflate speed MAE; identical cap across all methods.
        ro = clip_speed(np.asarray(rho_obs, dtype=float), self.speed_cap_mps)
        fin = ro[np.isfinite(ro)]
        if fin.size:
            self.rho_max = max(self.rho_max, float(fin.max()))
        for c in _CH:
            lp = np.asarray(log_p[c], dtype=float)
            if self._lp_floor is not None:
                # Shared floor: a matched density below it (including an exact
                # zero, log = -inf) is charged at the floor. NaN marks a channel
                # the method does not define and stays excluded via the finite
                # mask below.
                self.n_below[c] += int((m & (lp < self._lp_floor)).sum())
                lp = np.maximum(lp, self._lp_floor)
            fm = m & np.isfinite(lp)
            self.sum_lp[c] += float(lp[fm].sum())
            self.n_fm[c] += int(fm.sum())
        mr = np.asarray(mean_rho, dtype=float)
        vs = m & np.isfinite(mr)
        self.speed_ae_sum += float(np.abs(ro[vs] - mr[vs]).sum())
        self.speed_ae_n += int(vs.sum())

    def result(self):
        rho_max = max(self.rho_max, 1e-6)
        if self.rho_max_override > 0.0:
            if rho_max > self.rho_max_override + 1e-9:
                raise AssertionError(
                    f"observed (clipped) speed {rho_max:.3f} m/s exceeds the "
                    f"--rho-max override {self.rho_max_override:.3f}; floors "
                    f"would be inconsistent across runs")
            rho_max = self.rho_max_override
        floor = {
            "heading": -math.log(2.0 * math.pi),
            "speed": -math.log(rho_max),
            "joint": -math.log(2.0 * math.pi) - math.log(rho_max),
        }
        out = {
            "n_total": self.n_total,
            "n_matched": self.n_matched,
            "match_rate": (self.n_matched / self.n_total) if self.n_total else 0.0,
            "rho_max": rho_max,
            "density_floor": self.density_floor,
        }
        for c in _CH:
            out[f"below_floor_{c}"] = (
                self.n_below[c] / self.n_matched if self.n_matched else 0.0)
            if self.n_fm[c] == 0:
                out[f"mlpd_{c}"] = None
                out[f"mlpd_{c}_matched_only"] = None
                continue
            # mean over ALL detections: matched keep log-density, the rest get the floor.
            out[f"mlpd_{c}"] = (
                self.sum_lp[c] + (self.n_total - self.n_fm[c]) * floor[c]
            ) / self.n_total
            out[f"mlpd_{c}_matched_only"] = self.sum_lp[c] / self.n_fm[c]
        out["speed_mae_mps"] = (
            self.speed_ae_sum / self.speed_ae_n if self.speed_ae_n else None
        )
        return out


class CoverageStratifiedAccumulator:
    """MLPD stratified by per-voxel coverage (where evidence sharing acts).

    Coverage proxy = number of matched detections that land in a voxel across the
    whole test set, counted from the preds' x/y. Holds per-voxel
    (sum log_p_joint, count) so it is single-pass; the sparse/medium/dense split
    is applied at result() once every voxel's total count is known."""

    _OFF = 100000  # voxel-index offset so negatives pack into a positive key
    _STRIDE = 1000000

    def __init__(self, voxel_size: float = 0.4, thresholds=(10, 50)):
        self.vs = voxel_size
        self.lo, self.hi = thresholds
        self.sum_lp = {}   # voxel key -> sum of matched log_p_joint
        self.count = {}    # voxel key -> matched detection count

    def update(self, matched, xyz, log_p_joint):
        m = np.asarray(matched, dtype=bool)
        lp = np.asarray(log_p_joint, dtype=float)
        xy = np.asarray(xyz, dtype=float)
        v = m & np.isfinite(lp) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        if not v.any():
            return
        vx = np.floor(xy[v, 0] / self.vs).astype(np.int64) + self._OFF
        vy = np.floor(xy[v, 1] / self.vs).astype(np.int64) + self._OFF
        key = vx * self._STRIDE + vy
        uk, inv = np.unique(key, return_inverse=True)
        csum = np.zeros(uk.size); ccnt = np.zeros(uk.size, dtype=np.int64)
        np.add.at(csum, inv, lp[v])
        np.add.at(ccnt, inv, 1)
        for k, s, c in zip(uk.tolist(), csum.tolist(), ccnt.tolist()):
            self.sum_lp[k] = self.sum_lp.get(k, 0.0) + s
            self.count[k] = self.count.get(k, 0) + c

    def result(self):
        buckets = {"sparse": [0.0, 0], "medium": [0.0, 0], "dense": [0.0, 0]}
        for k, c in self.count.items():
            b = "sparse" if c < self.lo else ("medium" if c < self.hi else "dense")
            buckets[b][0] += self.sum_lp[k]
            buckets[b][1] += c
        out = {"lo": self.lo, "hi": self.hi}
        for b, (s, n) in buckets.items():
            out[f"mlpd_{b}"] = (s / n) if n > 0 else None
            out[f"n_{b}"] = n
        return out


class TemporalStratifiedAccumulator:
    """Joint MLPD stratified by time elapsed since session start. Bins
    detections by (t - session_start)/60 into the
    bins_min edges; session_start defaults to the first finite timestamp seen
    (== the global min for a time-sorted stream, as in the batch). Detections past
    the last edge are dropped, matching the batch's half-open buckets."""

    def __init__(self, session_start_s=None, bins_min=(0, 15, 30, 60, 120)):
        self.session_start = session_start_s
        self.edges = list(bins_min)
        self.nb = len(self.edges) - 1
        self.sum_lp = [0.0] * self.nb
        self.cnt = [0] * self.nb
        self.n_valid = 0

    def update(self, matched, t_s, log_p_joint):
        m = np.asarray(matched, dtype=bool)
        t = np.asarray(t_s, dtype=float)
        lp = np.asarray(log_p_joint, dtype=float)
        v = m & np.isfinite(t) & np.isfinite(lp)
        if not v.any():
            return
        tv = t[v]; lpv = lp[v]
        if self.session_start is None:
            self.session_start = float(tv.min())
        elapsed = (tv - self.session_start) / 60.0
        for i in range(self.nb):
            mask = (elapsed >= self.edges[i]) & (elapsed < self.edges[i + 1])
            self.sum_lp[i] += float(lpv[mask].sum())
            self.cnt[i] += int(mask.sum())
        self.n_valid += int(v.sum())

    def result(self):
        out = {"temporal_session_start_s": self.session_start,
               "temporal_n_valid": self.n_valid}
        for i in range(self.nb):
            lo, hi = self.edges[i], self.edges[i + 1]
            n = self.cnt[i]
            out[f"temporal_mlpd_{lo}_{hi}min"] = (self.sum_lp[i] / n) if n > 0 else None
            out[f"temporal_n_{lo}_{hi}min"] = n
        return out


class PhaseStratifiedAccumulator:
    """Joint MLPD folded by phase within each configured period, for datasets
    whose recurrence is at hour/day/week scale (ATC, HB), where the
    session-elapsed stratification is meaningless. Each detection's MLPD is
    binned by (t mod period) for every period; e.g. daily (86400 s, 24 bins) gives
    hour-of-day and weekly (604800 s, 7 bins) gives day-of-week. The per-bin MLPD
    shows how the forecast does at the phases where flow varies, which the
    time-averaged global MLPD washes out.

    The periods are a PARAMETER, set per dataset exactly like the NUDFT omegas:
    pass `phases = [(name, period_s, n_bins), ...]`. Phase is relative to the epoch
    (unix 0); the absolute offset is irrelevant since every method uses the same fold."""

    def __init__(self, phases):
        self.phases = [(str(n), float(p), int(b)) for (n, p, b) in (phases or [])]
        self.sum_lp = {n: np.zeros(b) for (n, p, b) in self.phases}
        self.cnt = {n: np.zeros(b, dtype=np.int64) for (n, p, b) in self.phases}

    def update(self, matched, t_s, log_p_joint):
        if not self.phases:
            return
        m = np.asarray(matched, dtype=bool)
        t = np.asarray(t_s, dtype=float)
        lp = np.asarray(log_p_joint, dtype=float)
        v = m & np.isfinite(t) & np.isfinite(lp)
        if not v.any():
            return
        tv = t[v]; lpv = lp[v]
        for name, P, B in self.phases:
            b = np.floor(np.mod(tv, P) / (P / B)).astype(np.int64)
            np.clip(b, 0, B - 1, out=b)
            np.add.at(self.sum_lp[name], b, lpv)
            np.add.at(self.cnt[name], b, 1)

    def result(self):
        out = {}
        for name, P, B in self.phases:
            s = self.sum_lp[name]; c = self.cnt[name]
            out[name] = {"period_s": P, "n_bins": B, "n": c.tolist(),
                         "mlpd": [float(s[i] / c[i]) if c[i] > 0 else None
                                  for i in range(B)]}
        return out


class LcMlpdDeltaAccumulator:
    """Per-loop-closure-event joint-MLPD delta (loop-closure robustness). For
    each LC event t_lc, the mean matched
    log_p_joint over [t_lc - W, t_lc) (pre) and [t_lc, t_lc + W] (post); the
    per-event delta is post - pre. Streams the per-window sums so no detection is
    stored. `lc_events_s` is supplied by the caller (from the pipeline's merge
    events / a sidecar); with no events, result() is a no-op."""

    def __init__(self, lc_events_s, window_s):
        self.events = [float(e) for e in (lc_events_s or [])]
        self.w = float(window_s)
        n = len(self.events)
        self.pre_sum = np.zeros(n); self.pre_n = np.zeros(n, dtype=np.int64)
        self.post_sum = np.zeros(n); self.post_n = np.zeros(n, dtype=np.int64)

    def update(self, matched, t_s, log_p_joint):
        if not self.events:
            return
        m = np.asarray(matched, dtype=bool)
        t = np.asarray(t_s, dtype=float)
        lp = np.asarray(log_p_joint, dtype=float)
        v = m & np.isfinite(t)          # batch masks on matched & finite t only
        if not v.any():
            return
        tv = t[v]; lpv = lp[v]
        for i, t_lc in enumerate(self.events):
            pre = (tv >= t_lc - self.w) & (tv < t_lc)
            post = (tv >= t_lc) & (tv <= t_lc + self.w)
            self.pre_sum[i] += float(lpv[pre].sum()); self.pre_n[i] += int(pre.sum())
            self.post_sum[i] += float(lpv[post].sum()); self.post_n[i] += int(post.sum())

    def result(self):
        per_event = []; deltas = []
        for i, t_lc in enumerate(self.events):
            npre, npost = int(self.pre_n[i]), int(self.post_n[i])
            if npre == 0 or npost == 0:
                per_event.append({"t_lc_s": t_lc, "n_pre": npre, "n_post": npost,
                                  "delta_mlpd": None})
                continue
            pre = self.pre_sum[i] / npre; post = self.post_sum[i] / npost
            d = post - pre; deltas.append(d)
            per_event.append({"t_lc_s": t_lc, "n_pre": npre, "n_post": npost,
                              "mlpd_pre": pre, "mlpd_post": post, "delta_mlpd": d})
        if deltas:
            da = np.asarray(deltas)
            mean_delta = float(da.mean()); frac = float((da > 0).mean())
        else:
            mean_delta = None; frac = None
        return {"lc_mean_delta_mlpd": mean_delta, "lc_n_events_evaluated": len(deltas),
                "lc_frac_improved": frac, "lc_window_s": self.w, "lc_per_event": per_event}


class CalibrationAccumulator:
    """Speed-channel calibration: ECE, 68/95 % interval coverage, sharpness.

    Per matched detection with a positive std: z=|rho_obs-mean_rho|/std,
    credible_mass=erf(z/sqrt2); empirical coverage at threshold p = fraction with
    credible_mass<=p. Accumulates a tally per threshold so no values are stored."""

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins
        self.thresholds = np.linspace(0.0, 1.0, n_bins + 1)[1:]
        self.counts = np.zeros(n_bins, dtype=np.int64)
        self.n = 0            # matched & std>0
        self.cov68 = 0
        self.cov95 = 0
        self.sum_2std = 0.0

    def _tally(self, cm, counts):
        # NaN residuals (non-finite rho/mean) give NaN credible mass; NaN <= p is
        # False in NumPy, exactly as in the batch path, so they count toward the
        # denominator but never toward coverage.
        counts += np.array([int((cm <= p).sum()) for p in self.thresholds],
                           dtype=np.int64)
        return int((cm <= 0.6827).sum()), int((cm <= 0.95).sum())

    def update(self, matched, rho_obs, mean_rho, std_rho):
        m = np.asarray(matched, dtype=bool)
        if not m.any():
            return
        ro = np.asarray(rho_obs, dtype=float)
        mr = np.asarray(mean_rho, dtype=float)
        sd = np.asarray(std_rho, dtype=float)
        # Mask: matched & finite std>0 (no rho/mean finiteness guard). A
        # non-finite residual gives NaN credible mass, which _tally counts
        # toward n but never toward coverage.
        v = m & np.isfinite(sd) & (sd > 0.0)
        if v.any():
            resid = np.abs(ro[v] - mr[v])
            self.n += int(v.sum())
            cm = erf(resid / sd[v] / _SQRT2)
            c68, c95 = self._tally(cm, self.counts)
            self.cov68 += c68; self.cov95 += c95
            self.sum_2std += float((2.0 * sd[v]).sum())

    def result(self):
        if self.n == 0:
            return {"speed_ece": None, "speed_coverage_68": None,
                    "speed_coverage_95": None, "speed_sharpness_mps": None,
                    "n_calibration": 0, "n_bins": self.n_bins}
        emp = self.counts / self.n
        return {"n_calibration": int(self.n), "n_bins": self.n_bins,
                "speed_ece": float(np.mean(np.abs(emp - self.thresholds))),
                "speed_coverage_68": self.cov68 / self.n,
                "speed_coverage_95": self.cov95 / self.n,
                "speed_sharpness_mps": self.sum_2std / self.n}


class CouplingAccumulator:
    """Streaming coupled joint MLPD + conditional-heading MAE on adjacent-voxel
    pairs, stratified by D_KL(pi_i ‖ pi_j) -- the flow-dependence metric: the
    learned coupling scored against treating the two voxels as independent.

    Pairs are produced in a single pass by a voxel-bucketed sliding window: each
    incoming detection is compared only to the <=8 neighbouring voxels' buffered
    detections still inside the time window, and only voxels that are endpoints
    of a coupling edge are ever buffered. Because the per-pair contribution is
    order-independent, the running means/win-rates equal the mean over every
    pair of detections within the window.

    Feed detections in non-decreasing time order (the ATC source and a t-sorted
    preds both satisfy this). With no coupling edges nothing is buffered and
    result() reports zero pairs.
    """

    _K = 8
    _MU_THETA = np.array([k * math.pi / 4.0 for k in range(_K)])
    _AGREE_RAD = math.pi / 4.0
    _KL_BOUNDS = (("low", 0.0, 0.2), ("medium", 0.2, 1.0), ("high", 1.0, float("inf")))

    def __init__(self, coupling_json, voxel_size=0.4, time_window_s=1.0,
                 sigma_theta=0.4, sigma_rho=0.3, nudft_order=5):
        import json as _json
        if isinstance(coupling_json, (str, bytes)):
            data = _json.loads(coupling_json) if coupling_json else {}
        else:
            data = coupling_json or {}
        if isinstance(data, list):
            edges_raw = [e for item in data for e in item.get("edges", [])]
            self.omegas = next((item["omegas"] for item in data if item.get("omegas")), [])
        else:
            edges_raw = data.get("edges", [])
            self.omegas = data.get("omegas", [])

        self.vs = voxel_size
        self.window = time_window_s
        self.var_theta = sigma_theta ** 2
        self.var_rho = sigma_rho ** 2
        self.nudft_order = nudft_order
        self._omega_arr = np.asarray(self.omegas, dtype=np.float64)

        # Edge lookup keyed by canonical (a, b) 3-tuple pair (lexicographically
        # smaller voxel first), plus KL(pi_a ‖ pi_b) per edge and the set of
        # (vx, vy) voxels that are endpoints of any edge (buffer filter). Each
        # edge's phi spectrum is pre-reduced to its top-`nudft_order` components
        # so phi(t) is a single vectorised numpy expression at pair time instead
        # of 64 per-pair Python _nudft_predict calls (the same formula).
        # Each edge gets an integer id (gid). `edge_gid` maps the canonical voxel
        # key to it; per-gid arrays hold the phi spectrum, the fall-back marginals
        # and KL, so the flush can gather them with fancy indexing.
        self.edge_gid = {}
        self.endpoints = set()
        O = self.nudft_order
        KK = self._K * self._K
        g0s, mags, oms, phs = [], [], [], []
        pi_a_fb, pi_b_fb, kls, v1p, v2p = [], [], [], [], []
        for edge in edges_raw:
            a = tuple(edge["a"]); b = tuple(edge["b"])
            key = (a, b) if a <= b else (b, a)
            if key in self.edge_gid:
                continue
            gid = len(g0s)
            self.edge_gid[key] = gid
            # The edge joins these two voxels; role a/b at pair time is by TIME
            # (earlier = role a), matching true_j_mlpd, NOT by which voxel -- so
            # pi_a/phi orientation follows the stored edge["a"]/nudft_phi order.
            v1p.append(self._pack(a[0], a[1])); v2p.append(self._pack(b[0], b[1]))
            self.endpoints.add((a[0], a[1]))
            self.endpoints.add((b[0], b[1]))
            pa = np.asarray(edge.get("pi_a", []), dtype=np.float64)
            pb = np.asarray(edge.get("pi_b", []), dtype=np.float64)
            # Fall-back marginals used only when a detection carries no slot
            # weights; NaN-filled (and later masked out) when the edge lacks them,
            # matching _score_pair's size guard.
            pa_ok = pa.size == self._K and pa.sum() > 1e-9
            pb_ok = pb.size == self._K and pb.sum() > 1e-9
            pi_a_fb.append(pa if pa_ok else np.full(self._K, np.nan))
            pi_b_fb.append(pb if pb_ok else np.full(self._K, np.nan))
            if pa_ok and pb_ok:
                pan = pa / pa.sum(); pbn = pb / pb.sum()
                kls.append(float(np.sum(pan * np.log((pan + 1e-30) / (pbn + 1e-30)))))
            else:
                kls.append(0.0)
            g0, mag, om, ph = self._precompute_phi(edge.get("nudft_phi", []))
            g0s.append(g0); mags.append(mag); oms.append(om); phs.append(ph)

        E = len(g0s)
        self._edge_g0 = np.asarray(g0s).reshape(E, KK) if E else np.empty((0, KK))
        self._edge_mag = np.asarray(mags).reshape(E, KK, O) if E else np.empty((0, KK, O))
        self._edge_om = np.asarray(oms).reshape(E, KK, O) if E else np.empty((0, KK, O))
        self._edge_ph = np.asarray(phs).reshape(E, KK, O) if E else np.empty((0, KK, O))
        self._edge_pi_a_fb = np.asarray(pi_a_fb).reshape(E, self._K) if E else np.empty((0, self._K))
        self._edge_pi_b_fb = np.asarray(pi_b_fb).reshape(E, self._K) if E else np.empty((0, self._K))
        self._edge_kl = np.asarray(kls) if E else np.empty(0)
        self._edge_v1pack = np.asarray(v1p, dtype=np.int64) if E else np.empty(0, dtype=np.int64)
        self._edge_v2pack = np.asarray(v2p, dtype=np.int64) if E else np.empty(0, dtype=np.int64)
        # Kept for the compatibility of external checks (e.g. `if cpl.edge_lookup`).
        self.edge_lookup = self.edge_gid

        # Packed int64 keys of the endpoint voxels, for a vectorised isin()
        # pre-filter so only detections that can form an edge are ever buffered.
        self._endpoint_keys = np.array(
            sorted(self._pack(vx, vy) for (vx, vy) in self.endpoints),
            dtype=np.int64) if self.endpoints else np.empty(0, dtype=np.int64)

        # Block-buffer state. Detections are accumulated (vectorised, per chunk)
        # until >= _block_n, then a per-edge windowed join emits all pairs whose
        # LATER endpoint falls in the block; the last `window` seconds are carried
        # over so cross-block pairs are not missed (and not double-counted: a pair
        # is skipped only if BOTH endpoints are carry-over, i.e. already emitted).
        self._block_n = 2_000_000
        self._score_batch = 64_000   # pairs per vectorised _score() call (bounds the
                                     # (P,64,O) phi transient to a few hundred MB)
        self._buf = []               # list of per-chunk dicts (concatenated at flush)
        self._buf_n = 0
        self._seq = 0                # global arrival counter = batch time order
        self._carry = None           # carry-over detections dict for the next block

        # Running accumulators (aggregate + per-KL-stratum + disagree subset
        # + same/cross-agent strata when the source supplies track ids).
        self._agg = self._new_bucket()
        self._strata = {s: self._new_bucket() for s, _, _ in self._KL_BOUNDS}
        self._disagree = self._new_bucket()
        self._same_agent = self._new_bucket()
        self._cross_agent = self._new_bucket()
        self._n_id_unknown = 0
        self._n_pairs = 0

    @staticmethod
    def _new_bucket():
        # sum_coupled_lp, sum_marginal_lp, n_jmlpd, jmlpd_wins,
        # sum_cond_mae, sum_base_mae, n_cond, cond_wins.
        return {"c": 0.0, "m": 0.0, "n": 0, "jw": 0,
                "cm": 0.0, "bm": 0.0, "nc": 0, "cw": 0}

    _PACK_OFF = 1 << 20
    _PACK_STRIDE = 1 << 40

    @classmethod
    def _pack(cls, vx, vy):
        """Pack a (vx, vy) voxel into a single int64 (handles negatives)."""
        return (int(vx) + cls._PACK_OFF) * cls._PACK_STRIDE + (int(vy) + cls._PACK_OFF)

    @staticmethod
    def _vkey(vx, vy):
        return (int(vx), int(vy))  # buffer-dict key; tuples handle negative indices

    def _precompute_phi(self, nudft_phi):
        """Reduce an edge's K*K NUDFT-phi predictors to fixed-shape arrays so
        phi(t) = gamma0 + 2*sum_m mag*cos(omega*t + phase) evaluates in one numpy
        pass. Selection of the top-`nudft_order` magnitude components and the DC
        fallback (before min_observations, or no components) match
        the Kairos spectral predictor at a fixed order (NUDFTModel::predict);
        padded rows carry mag=0."""
        KK = self._K * self._K
        O = self.nudft_order
        n_omega = self._omega_arr.size
        g0 = np.zeros(KK); mag = np.zeros((KK, O)); om = np.zeros((KK, O)); ph = np.zeros((KK, O))
        for idx in range(min(KK, len(nudft_phi))):
            nd = nudft_phi[idx]
            g0[idx] = nd.get("gamma0", 0.0)
            n = nd.get("n", 0); min_obs = nd.get("min_observations", 1)
            gammas = nd.get("gammas", [])
            if n < min_obs or not gammas or O <= 0:
                continue
            comps = []
            for i, g in enumerate(gammas):
                if i < n_omega:
                    c = complex(g["real"], g["imag"])
                    comps.append((abs(c), self._omega_arr[i], np.angle(c)))
            comps.sort(key=lambda x: -x[0])
            for j, (m_, o_, p_) in enumerate(comps[:O]):
                mag[idx, j] = m_; om[idx, j] = o_; ph[idx, j] = p_
        return g0, mag, om, ph

    def _gauss_vec(self, theta, rho):
        """Per-slot unnormalised Gaussian density (mu_rho = 0) for an array of
        observations -> (P, K). Same formula as the batch gaussian_density."""
        d = np.abs(theta[:, None] - self._MU_THETA[None, :])
        d = np.minimum(d, 2.0 * math.pi - d)          # circular distance in [0, pi]
        return np.exp(-0.5 * (d ** 2 / self.var_theta + (rho[:, None] ** 2) / self.var_rho))

    def _bucket_add(self, b, mask, lp_c, lp_m, cerr, berr, cvalid):
        """Add a masked subset of the flushed pairs into one accumulator bucket.
        `mask` selects pairs; `cvalid`/`cerr`/`berr` are per-DIRECTION (2*P) with
        `mask` broadcast across the two directions."""
        if not mask.any():
            return
        b["c"] += float(lp_c[mask].sum())
        b["m"] += float(lp_m[mask].sum())
        b["n"] += int(mask.sum())
        b["jw"] += int((lp_c[mask] > lp_m[mask]).sum())
        dmask = np.concatenate([mask, mask]) & cvalid
        b["cm"] += float(cerr[dmask].sum())
        b["bm"] += float(berr[dmask].sum())
        b["nc"] += int(dmask.sum())
        b["cw"] += int((cerr[dmask] < berr[dmask]).sum())

    def _circ_deg(self, pred_k, target):
        d = np.abs(self._MU_THETA[pred_k] - target)
        d = np.minimum(d, 2.0 * math.pi - d)
        return np.degrees(d)

    def _resolve_sw(self, slot_weights, idx):
        """Slot weights for the candidate rows `idx` -> (len(idx), K), with NaN
        rows where a detection has no usable weights (wrong length or sum<=0);
        those fall back to the edge marginals at join time. Fast path for a clean
        (N, K) array; per-row fallback for the ragged list the pipeline returns."""
        K = self._K
        if slot_weights is None:
            return np.full((idx.size, K), np.nan)
        if isinstance(slot_weights, np.ndarray) and slot_weights.ndim == 2 and slot_weights.shape[1] == K:
            sw = slot_weights[idx].astype(np.float64, copy=True)
            bad = ~np.isfinite(sw).all(1) | (sw.sum(1) <= 0)
            sw[bad] = np.nan
            return sw
        out = np.full((idx.size, K), np.nan)
        for r, i in enumerate(idx.tolist()):
            row = slot_weights[i]
            if len(row) == K:
                a = np.asarray(row, dtype=np.float64)
                if np.isfinite(a).all() and a.sum() > 0:
                    out[r] = a
        return out

    def _score(self, gid, pia, pib, tha, rha, thb, rhb, ta, ida=None, idb=None):
        """Vectorised scoring of a batch of pairs (role a = the earlier detection).
        Mirrors true_j_mlpd's per-pair body applied to the whole batch at once.
        `ida`/`idb` are optional per-pair track ids for the same/cross-agent
        strata (-1 = unknown)."""
        if gid.size == 0:
            return
        K = self._K
        # Normalise marginals; drop pairs whose marginals are absent/degenerate
        # (NaN fall-back or ~zero sum), matching _score_pair's size/sum guards.
        sa = pia.sum(1); sb = pib.sum(1)
        keep = np.isfinite(sa) & np.isfinite(sb) & (sa > 1e-9) & (sb > 1e-9)
        if not keep.any():
            return
        gid = gid[keep]; ta = ta[keep]
        if ida is not None:
            ida = ida[keep]; idb = idb[keep]
        pia = pia[keep] / sa[keep, None]; pib = pib[keep] / sb[keep, None]
        n_i = self._gauss_vec(tha[keep], rha[keep])       # (P, K)
        n_j = self._gauss_vec(thb[keep], rhb[keep])
        tha = tha[keep]; thb = thb[keep]

        # phi(t) = gamma0 + 2*sum_m mag*cos(omega*t + phase), gathered per edge.
        phi = (self._edge_g0[gid]
               + 2.0 * np.sum(self._edge_mag[gid] * np.cos(self._edge_om[gid] * ta[:, None, None]
                                                           + self._edge_ph[gid]), axis=2))
        phi = phi.reshape(-1, K, K)
        exp_phi = np.exp(phi)                              # (P, K, K)

        pin_i = pia * n_i                                 # (P, K)
        pin_j = pib * n_j
        w = pin_i[:, :, None] * pin_j[:, None, :] * exp_phi
        Z = (pia[:, :, None] * pib[:, None, :] * exp_phi).sum((1, 2))
        coupled = np.divide(w.sum((1, 2)), Z, out=np.zeros_like(Z), where=Z >= 1e-30)
        marginal = pin_i.sum(1) * pin_j.sum(1)
        good = (Z >= 1e-30) & (coupled > 0) & (marginal > 0)
        if not good.any():
            return
        gid = gid[good]
        pia = pia[good]; pib = pib[good]; pin_i = pin_i[good]; pin_j = pin_j[good]
        n_i = n_i[good]; n_j = n_j[good]; phi = phi[good]; exp_phi = exp_phi[good]
        tha = tha[good]; thb = thb[good]
        if ida is not None:
            ida = ida[good]; idb = idb[good]
        lp_c = np.log(coupled[good]); lp_m = np.log(marginal[good])
        P = lp_c.size
        ar = np.arange(P)

        # Conditional-heading errors, both directions, vs the marginal prediction.
        def _dir(pa, pb, ni, phi_row_sel, target):
            # phi_row_sel: (P, K) coupling row selected by the source slot.
            k_src = np.argmax(pa * ni, axis=1)             # source dominant slot
            p_cond = pb * np.exp(phi_row_sel(k_src))
            s = p_cond.sum(1)
            valid = s >= 1e-30
            with np.errstate(invalid="ignore"):
                p_cond = np.divide(p_cond, s[:, None], out=np.zeros_like(p_cond),
                                   where=s[:, None] >= 1e-30)
            k_cond = np.argmax(p_cond, axis=1)
            k_marg = np.argmax(pb, axis=1)
            return self._circ_deg(k_cond, target), self._circ_deg(k_marg, target), valid

        # i->j: source i (pi_a, n_i), phi row k_i -> phi[p, k_i, :]; target thb.
        cerr_ij, berr_ij, v_ij = _dir(pia, pib, n_i, lambda k: phi[ar, k, :], thb)
        # j->i: source j (pi_b, n_j), phi.T row k_j -> phi[p, :, k_j]; target tha.
        cerr_ji, berr_ji, v_ji = _dir(pib, pia, n_j, lambda k: phi[ar, :, k], tha)
        cerr = np.concatenate([cerr_ij, cerr_ji])
        berr = np.concatenate([berr_ij, berr_ji])
        cvalid = np.concatenate([v_ij, v_ji])

        kl = self._edge_kl[gid]
        d_obs = np.abs(tha - thb)
        d_obs = np.minimum(d_obs, 2.0 * math.pi - d_obs)
        disagree = d_obs > self._AGREE_RAD

        all_mask = np.ones(P, dtype=bool)
        self._bucket_add(self._agg, all_mask, lp_c, lp_m, cerr, berr, cvalid)
        for sname, lo, hi in self._KL_BOUNDS:
            self._bucket_add(self._strata[sname], (kl >= lo) & (kl < hi),
                             lp_c, lp_m, cerr, berr, cvalid)
        self._bucket_add(self._disagree, disagree, lp_c, lp_m, cerr, berr, cvalid)
        if ida is not None:
            known = (ida >= 0) & (idb >= 0)
            self._bucket_add(self._same_agent, known & (ida == idb),
                             lp_c, lp_m, cerr, berr, cvalid)
            self._bucket_add(self._cross_agent, known & (ida != idb),
                             lp_c, lp_m, cerr, berr, cvalid)
            self._n_id_unknown += int((~known).sum())
        self._n_pairs += P

    def update(self, matched, t_s, xyz, theta, rho, slot_weights, ids=None):
        """Buffer a time-ordered chunk of detections. `slot_weights` is an (N, K)
        array or a ragged list; rows that are not length-K or sum<=0 fall back to
        the edge marginals at join time. `ids` is an optional per-detection track
        id (-1/None = unknown) enabling the same/cross-agent strata. Pairs are
        formed later, per-block, by a vectorised windowed join -- no per-pair
        Python."""
        if not self.edge_gid or self._endpoint_keys.size == 0:
            return
        m = np.asarray(matched, dtype=bool)
        t = np.asarray(t_s, dtype=float)
        xy = np.asarray(xyz, dtype=float)
        th = np.asarray(theta, dtype=float)
        rh = np.asarray(rho, dtype=float)
        vs = self.vs

        # Vectorised pre-filter: only matched, finite detections in an edge-endpoint
        # voxel can ever form a coupling pair.
        valid = (m & np.isfinite(t) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
                 & np.isfinite(th) & np.isfinite(rh))
        if not valid.any():
            return
        vxa = np.floor(xy[:, 0] / vs).astype(np.int64)
        vya = np.floor(xy[:, 1] / vs).astype(np.int64)
        vkey = (vxa + self._PACK_OFF) * self._PACK_STRIDE + (vya + self._PACK_OFF)
        cand = valid & np.isin(vkey, self._endpoint_keys)
        idx = np.flatnonzero(cand)  # ascending -> chunk time order preserved
        if idx.size == 0:
            return
        n = idx.size
        if ids is None:
            id_arr = np.full(n, -1, dtype=np.int64)
        else:
            id_arr = np.asarray(ids, dtype=np.int64)[idx]
        self._buf.append({
            "t": t[idx], "th": th[idx], "rho": rh[idx],
            "sw": self._resolve_sw(slot_weights, idx), "vk": vkey[idx],
            "id": id_arr,
            "seq": self._seq + np.arange(n, dtype=np.int64),
            "carry": np.zeros(n, dtype=bool),
        })
        self._seq += n
        self._buf_n += n
        if self._buf_n >= self._block_n:
            self._block_flush()

    def _block_flush(self, final=False):
        """Form and score all pairs whose LATER endpoint lies in the current block,
        via a per-edge windowed join. The last `window` seconds are retained as
        carry-over for the next block; a pair is skipped only if BOTH endpoints are
        carry-over (already emitted last block), so every pair is scored once."""
        parts = ([self._carry] if self._carry is not None else []) + self._buf
        self._buf = []; self._buf_n = 0
        if not parts:
            return
        cat = {k: np.concatenate([p[k] for p in parts]) for k in
               ("t", "th", "rho", "vk", "id", "seq", "carry")}
        sw_u = np.concatenate([p["sw"] for p in parts])   # unsorted
        M = cat["t"].size

        # Group by voxel (stable sort keeps within-voxel arrival/time order).
        order = np.argsort(cat["vk"], kind="stable")
        t = cat["t"][order]; th = cat["th"][order]; rho = cat["rho"][order]
        vk = cat["vk"][order]; seq = cat["seq"][order]; carry = cat["carry"][order]
        det_id = cat["id"][order]
        sws = sw_u[order]                                 # sorted, used in the join
        uvk, starts = np.unique(vk, return_index=True)
        ends = np.append(starts[1:], M)
        vmap = {int(k): (int(s), int(e)) for k, s, e in zip(uvk, starts, ends)}

        win = self.window
        batch = {k: [] for k in ("gid", "ea", "el")}
        pending = 0

        def _emit():
            nonlocal pending
            if not batch["gid"]:
                return
            gid = np.concatenate(batch["gid"])
            ea = np.concatenate(batch["ea"]); el = np.concatenate(batch["el"])
            pia = sws[ea].copy(); pib = sws[el].copy()
            # A matched detection without usable slot weights drops its pairs
            # (the NaN rows fail _score's finite/sum guard), matching the
            # reference per-pair scorer; substituting the edge's training-time
            # marginals here would score a different quantity for those rows.
            self._score(gid, pia, pib, th[ea], rho[ea], th[el], rho[el], t[ea],
                        det_id[ea], det_id[el])
            batch["gid"] = []; batch["ea"] = []; batch["el"] = []
            pending = 0

        for gid in range(self._edge_v1pack.size):
            r1 = vmap.get(int(self._edge_v1pack[gid]))
            if r1 is None:
                continue
            r2 = vmap.get(int(self._edge_v2pack[gid]))
            if r2 is None:
                continue
            s1, e1 = r1; s2, e2 = r2
            t1 = t[s1:e1]; t2 = t[s2:e2]
            lo = np.searchsorted(t2, t1 - win, "left")
            hi = np.searchsorted(t2, t1 + win, "right")
            counts = hi - lo
            tot = int(counts.sum())
            if tot == 0:
                continue
            i1 = np.repeat(np.arange(e1 - s1), counts)
            i2 = (np.arange(tot) - np.repeat(np.cumsum(counts) - counts, counts)
                  + np.repeat(lo, counts))
            g1 = s1 + i1; g2 = s2 + i2
            both_carry = carry[g1] & carry[g2]
            if both_carry.any():
                keepp = ~both_carry
                g1 = g1[keepp]; g2 = g2[keepp]
                if g1.size == 0:
                    continue
            # Role by arrival order: earlier (smaller seq) is role a.
            a_is_1 = seq[g1] <= seq[g2]
            ea = np.where(a_is_1, g1, g2)
            el = np.where(a_is_1, g2, g1)
            batch["gid"].append(np.full(ea.size, gid, dtype=np.int64))
            batch["ea"].append(ea); batch["el"].append(el)
            pending += ea.size
            if pending >= self._score_batch:
                _emit()
        _emit()

        # Retain the last `window` seconds as carry-over for the next block
        # (computed on the unsorted arrays so sw_u indexing is consistent).
        if final:
            self._carry = None
        else:
            tmax = float(cat["t"].max())
            cm = cat["t"] > (tmax - win)
            self._carry = {"t": cat["t"][cm], "th": cat["th"][cm], "rho": cat["rho"][cm],
                           "sw": sw_u[cm], "vk": cat["vk"][cm], "id": cat["id"][cm],
                           "seq": cat["seq"][cm],
                           "carry": np.ones(int(cm.sum()), dtype=bool)}

    @staticmethod
    def _finalize(b):
        if b["n"] == 0:
            return {"j_mlpd": None, "baseline_mlpd": None, "j_mlpd_gain": None,
                    "j_mlpd_win_rate": None, "cond_heading_mae_deg": None,
                    "cond_heading_mae_baseline_deg": None, "cond_heading_mae_gain_deg": None,
                    "cond_heading_win_rate": None, "n_pairs": 0}
        c = b["c"] / b["n"]; mm = b["m"] / b["n"]
        cm = (b["cm"] / b["nc"]) if b["nc"] else None
        bm = (b["bm"] / b["nc"]) if b["nc"] else None
        return {
            "j_mlpd": c,
            "baseline_mlpd": mm,
            "j_mlpd_gain": c - mm,
            "j_mlpd_win_rate": b["jw"] / b["n"],
            "cond_heading_mae_deg": cm,
            "cond_heading_mae_baseline_deg": bm,
            "cond_heading_mae_gain_deg": (bm - cm) if (cm is not None and bm is not None) else None,
            "cond_heading_win_rate": (b["cw"] / b["nc"]) if b["nc"] else None,
            "n_pairs": b["n"],
        }

    def result(self):
        self._block_flush(final=True)  # join+score any buffered detections
        agg = self._finalize(self._agg)
        # Match the true_j_mlpd public key names on the aggregate.
        out = {
            "true_j_mlpd": agg["j_mlpd"],
            "true_j_mlpd_baseline_mlpd": agg["baseline_mlpd"],
            "true_j_mlpd_gain": agg["j_mlpd_gain"],
            "j_mlpd_win_rate": agg["j_mlpd_win_rate"],
            "cond_heading_mae_deg": agg["cond_heading_mae_deg"],
            "cond_heading_mae_baseline_deg": agg["cond_heading_mae_baseline_deg"],
            "cond_heading_mae_gain_deg": agg["cond_heading_mae_gain_deg"],
            "cond_heading_win_rate": agg["cond_heading_win_rate"],
            "true_j_mlpd_n_pairs": agg["n_pairs"],
            "agent_strata": {
                "same": self._finalize(self._same_agent),
                "cross": self._finalize(self._cross_agent),
                "n_pairs_id_unknown": self._n_id_unknown,
            },
            "disagree_subset": {"j_mlpd_gain": self._finalize(self._disagree)["j_mlpd_gain"],
                                "cond_mae_gain_deg": self._finalize(self._disagree)["cond_heading_mae_gain_deg"],
                                "n_pairs": self._disagree["n"]},
            "kl_strata": {},
        }
        for s, _, _ in self._KL_BOUNDS:
            f = self._finalize(self._strata[s])
            out["kl_strata"][s] = {"j_mlpd_gain": f["j_mlpd_gain"],
                                   "cond_mae_gain_deg": f["cond_heading_mae_gain_deg"],
                                   "n_pairs": f["n_pairs"]}
        return out


# ------------------------------------------------------------------ weight cal
def _log_binom_coeff(n, k):
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def _beta_binomial_pmf_cdf(n, a, b):
    """pmf and cdf of the beta-binomial count on 0..n (renormalized so the
    discrete PIT below is exact under floating point)."""
    k = np.arange(n + 1, dtype=float)
    logpmf = _log_binom_coeff(float(n), k) + betaln(k + a, n - k + b) - betaln(a, b)
    pmf = np.exp(logpmf - logpmf.max())
    pmf /= pmf.sum()
    return pmf, np.cumsum(pmf)


def _binomial_pmf_cdf(n, p):
    p = min(max(float(p), 1.0e-12), 1.0 - 1.0e-12)
    k = np.arange(n + 1, dtype=float)
    logpmf = _log_binom_coeff(float(n), k) + k * math.log(p) + (n - k) * math.log1p(-p)
    pmf = np.exp(logpmf - logpmf.max())
    pmf /= pmf.sum()
    return pmf, np.cumsum(pmf)


def _randomized_pit(pmf, cdf, x, u):
    """Randomized probability integral transform of statistic x against the
    discrete predictive: F(ceil(x)-1) + u * pmf(ceil(x)). Exactly Uniform(0,1)
    under the null for integer x; the soft statistic is rounded to its nearest
    count before this call (interpolating with frac(x) instead of a seeded u
    degenerates to the super-uniform unrandomized PIT precisely when
    responsibilities are peaked, and ceil misassigns the count bin whenever
    the soft contamination pushes x just past an integer)."""
    n = pmf.size - 1
    k_hi = int(math.ceil(x - 1.0e-9))
    k_hi = max(0, min(n, k_hi))
    lo = float(cdf[k_hi - 1]) if k_hi >= 1 else 0.0
    return min(1.0, max(0.0, lo + u * float(pmf[k_hi])))


def _stable_uniform(*key):
    """Deterministic uniform in [0,1) from a hashed key. A per-sample value
    that depends only on the sample's identity, never on arrival order, so
    results are invariant to the source's chunking."""
    h = hashlib.blake2b(",".join(str(k) for k in key).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") / 2.0 ** 64


class _CalTally:
    """Threshold tally for one stratum: per-threshold PIT counts, 68/95
    coverage, sharpness sum. Mirrors CalibrationAccumulator's tallying."""

    __slots__ = ("counts", "cov68", "cov95", "n", "sum_width")

    def __init__(self, n_bins):
        self.counts = np.zeros(n_bins, dtype=np.int64)
        self.cov68 = 0
        self.cov95 = 0
        self.n = 0
        self.sum_width = 0.0

    def add(self, cm, thresholds, width=0.0):
        self.counts += (cm <= thresholds).astype(np.int64)
        self.cov68 += int(cm <= 0.6827)
        self.cov95 += int(cm <= 0.95)
        self.n += 1
        self.sum_width += width

    def result(self, thresholds):
        if self.n == 0:
            return {"ece": None, "coverage_68": None, "coverage_95": None,
                    "sharpness_share": None, "n": 0}
        emp = self.counts / self.n
        return {"ece": float(np.mean(np.abs(emp - thresholds))),
                "coverage_68": self.cov68 / self.n,
                "coverage_95": self.cov95 / self.n,
                "sharpness_share": self.sum_width / self.n,
                "n": int(self.n)}


class WeightCalibrationAccumulator:
    """Mixing-weight calibration: coverage of a beta-binomial predictive for
    the window share of soft slot responsibilities, per (window, slot).

    The observable is the mean of the per-crossing soft responsibilities over
    a `dt_win` window at one voxel: crossings are rebuilt eval-side with the
    trainer's own visit rules (new visit on voxel change, silence > gap, span
    > max; vote = the visit's mean responsibility vector stamped at its
    midpoint; id-less detections vote per frame), so the model is tested on
    the exact quantity its weights are accumulated from. The predictive folds
    the propagated weight-parameter variance (a Beta on the underlying
    fraction, moment-matched, capped below the Bernoulli bound) with a
    binomial sampling layer, and each sample's credible mass is the seeded
    randomized PIT of the share against that predictive. Windows are
    non-overlapping, tiled per voxel within each stream day (day = gap >
    `day_break_s`), scored once their trailing half has streamed past;
    windows the day ends inside are dropped as truncated.

    Stated approximations (see the metric doc): the binomial layer upper-
    bounds the soft-vote scatter (conservative); the per-slot parameter
    variances are treated as independent though the votes couple them; the
    window predictive mean conditions on the realized vote times; within-
    window group correlation is not modeled -- it is sized by the burstiness
    strata and the cluster bootstrap, not removed.

    Two variants share every window: `soft` (primary) and `hard` (argmax-vote
    integer counts -- differs from soft only through the soft-vote layer, so
    agreement bounds that layer's conservatism and nothing else).

    Raises on NaN weight stds: a NaN marks a loaded flow state without
    variance tracking, which must abort the metric rather than score as
    zero-width intervals.
    """

    _SLOTS = 8

    def __init__(self, dt_win, min_count=5, voxel_size=0.4,
                 visit_gap_s=2.0, visit_max_s=10.0, day_break_s=4.0 * 3600.0,
                 n_bins=20, seed=0, var_cap_frac=0.9,
                 count_edges=(10, 20), evidence_edges=(50, 200),
                 burst_gap_s=2.0, burst_frac=0.3,
                 bootstrap_b=500, ece_null_m=200):
        if dt_win <= 0:
            raise ValueError("dt_win must be positive")
        self.dt = float(dt_win)
        self.min_count = int(min_count)
        self.vs = float(voxel_size)
        self.gap = float(visit_gap_s)
        self.vmax = float(visit_max_s)
        self.day_break = float(day_break_s)
        self.n_bins = int(n_bins)
        self.thresholds = np.linspace(0.0, 1.0, n_bins + 1)[1:]
        self.seed = int(seed)
        self.cap_frac = float(var_cap_frac)
        self.count_edges = tuple(count_edges)
        self.evidence_edges = tuple(evidence_edges)
        self.burst_gap = float(burst_gap_s)
        self.burst_frac = float(burst_frac)
        self.bootstrap_b = int(bootstrap_b)
        self.ece_null_m = int(ece_null_m)

        # Visit state: pid -> [vx, vy, t_first, t_last, n_frames, sum_resp(8),
        # n_matched, sum_pi(8), sum_var(8), clamp_any(8), order_max(8), n_cross]
        self._visits = {}
        # Window state: (vx, vy, w_idx) -> dict, within the current day.
        self._windows = {}
        self._day_idx = -1
        self._day_anchor = None
        self._day_last_t = None
        self._last_t = None

        K = self._SLOTS
        self._tallies = {v: _CalTally(n_bins) for v in ("soft", "hard")}
        self._strata = {
            "count": {}, "evidence": {}, "slot": {}, "clamp": {},
            "cap": {}, "order": {}, "burst": {},
        }
        # (vx, vy, day) cluster tallies for the bootstrap, soft variant.
        self._clusters = {}
        # Per-voxel coverage for the voxel-balanced companion (soft variant).
        self._voxels = {}
        # Parameter-share decomposition sums, overall and per evidence bin.
        self._pshare = {}
        # Bookkeeping.
        self._c = {"windows_scored": 0, "windows_below_min": 0,
                   "windows_truncated": 0, "votes_matched": 0,
                   "votes_unmatched": 0, "votes_idless": 0,
                   "late_votes_dropped": 0, "samples_capped": 0,
                   "samples_clamped": 0, "samples_order0": 0,
                   "samples_degenerate": 0, "samples_total": 0}
        self._K = K

    # ------------------------------------------------------------- streaming
    def update(self, t_s, xyz, ids, matched, resp,
               slot_weights, slot_stds, slot_clamped, slot_orders, n_crossings):
        """Feed one chunk. `resp` is (N, 8) observed responsibilities (model-
        independent); the slot_* arguments are the score API's ragged per-slot
        vectors (empty rows where unmatched); `n_crossings` the matched cell's
        training evidence."""
        t = np.asarray(t_s, float)
        if t.size == 0:
            return
        xy = np.asarray(xyz, float)
        vx = np.floor(xy[:, 0] / self.vs).astype(np.int64)
        vy = np.floor(xy[:, 1] / self.vs).astype(np.int64)
        m = np.asarray(matched, bool)
        resp = np.asarray(resp, float)
        ncr = np.asarray(n_crossings, np.int64)

        for i in range(t.size):
            ti = float(t[i])
            if self._last_t is None or ti - self._last_t > self.day_break:
                self._end_day()
                self._day_idx += 1
                self._day_anchor = ti
            self._last_t = ti
            self._day_last_t = ti

            row_pi = row_var = row_clamp = row_order = None
            if m[i]:
                w = np.asarray(slot_weights[i], float)
                if w.size == self._K:
                    sd = np.asarray(slot_stds[i], float)
                    if np.isnan(sd).any():
                        raise ValueError(
                            "weight std is NaN: the loaded flow state carries no "
                            "weight-variance state; retrain it with this version "
                            "of Kairos before scoring this metric")
                    row_pi = w
                    row_var = sd * sd
                    row_clamp = np.asarray(slot_clamped[i], np.int64)
                    row_order = np.asarray(slot_orders[i], np.int64)

            pid = int(ids[i]) if ids is not None else -1
            if pid < 0:
                # id-less fallback: one vote per frame, as in training.
                self._c["votes_idless"] += 1
                self._cast_vote(int(vx[i]), int(vy[i]), ti, resp[i],
                                row_pi, row_var, row_clamp, row_order,
                                int(ncr[i]))
                continue

            st = self._visits.get(pid)
            if st is not None and (st[0] != vx[i] or st[1] != vy[i]
                                   or ti - st[3] > self.gap
                                   or ti - st[2] > self.vmax):
                self._close_visit(st)
                st = None
            if st is None:
                st = [int(vx[i]), int(vy[i]), ti, ti, 0, np.zeros(self._K),
                      0, np.zeros(self._K), np.zeros(self._K),
                      np.zeros(self._K, np.int64), np.zeros(self._K, np.int64), 0]
                self._visits[pid] = st
            st[3] = ti
            st[4] += 1
            st[5] += resp[i]
            if row_pi is not None:
                st[6] += 1
                st[7] += row_pi
                st[8] += row_var
                st[9] |= row_clamp
                st[10] = np.maximum(st[10], row_order)
                st[11] = max(st[11], int(ncr[i]))

        # Score windows whose trailing half has fully streamed past. Votes are
        # stamped at visit midpoints, so a vote can arrive up to ~visit_max_s
        # behind the stream clock; the slack keeps their windows open.
        self._close_ripe_windows(float(t[-1]) - self.vmax)

    def _cast_vote(self, vxi, vyi, t_mid, resp_mean,
                   pi, var, clamp, order, ncross):
        matched = pi is not None
        if matched:
            self._c["votes_matched"] += 1
        else:
            self._c["votes_unmatched"] += 1
        w_idx = int(math.floor((t_mid - self._day_anchor) / self.dt))
        key = (vxi, vyi, w_idx)
        win = self._windows.get(key)
        if win is None:
            end = self._day_anchor + (w_idx + 1) * self.dt
            if self._last_t is not None and end < self._last_t - 2.0 * self.vmax:
                self._c["late_votes_dropped"] += 1
                return
            win = {"end": end, "n": 0, "total": 0,
                   "resp": np.zeros(self._K), "pi": np.zeros(self._K),
                   "var": np.zeros(self._K),
                   "clamp": np.zeros(self._K, np.int64),
                   "order": np.zeros(self._K, np.int64),
                   "hard": np.zeros(self._K, np.int64),
                   "ncross": 0, "prev_t": None, "close_gaps": 0}
            self._windows[key] = win
        win["total"] += 1
        if not matched:
            return
        if win["prev_t"] is not None and t_mid - win["prev_t"] < self.burst_gap:
            win["close_gaps"] += 1
        win["prev_t"] = t_mid
        win["n"] += 1
        win["resp"] += resp_mean
        win["pi"] += pi
        win["var"] += var
        win["clamp"] |= clamp
        win["order"] = np.maximum(win["order"], order)
        win["hard"][int(np.argmax(resp_mean))] += 1
        win["ncross"] = max(win["ncross"], ncross)

    def _close_visit(self, st):
        if st[4] == 0:
            return
        t_mid = 0.5 * (st[2] + st[3])
        resp_mean = st[5] / st[4]
        if st[6] > 0:
            self._cast_vote(st[0], st[1], t_mid, resp_mean,
                            st[7] / st[6], st[8] / st[6], st[9], st[10], st[11])
        else:
            self._cast_vote(st[0], st[1], t_mid, resp_mean,
                            None, None, None, None, 0)

    def _close_ripe_windows(self, before_t):
        ripe = [k for k, w in self._windows.items() if w["end"] < before_t]
        for k in ripe:
            self._score_window(k, self._windows.pop(k))

    def _end_day(self):
        """Close the ending day: flush visits (their votes land in this day's
        windows), then score every window the day actually completed and drop
        the ones it ended inside."""
        for st in list(self._visits.values()):
            self._close_visit(st)
        self._visits.clear()
        if self._day_last_t is not None:
            for k, w in list(self._windows.items()):
                if w["end"] <= self._day_last_t:
                    self._score_window(k, w)
                else:
                    self._c["windows_truncated"] += 1
            self._windows.clear()
        self._day_last_t = None

    # --------------------------------------------------------------- scoring
    @staticmethod
    def _bin_label(value, edges, prefix):
        for e in edges:
            if value < e:
                return f"{prefix}<{e}"
        return f"{prefix}>={edges[-1]}" if edges else f"{prefix}all"

    def _stratum(self, family, label):
        d = self._strata[family]
        tal = d.get(label)
        if tal is None:
            tal = _CalTally(self.n_bins)
            d[label] = tal
        return tal

    def _score_window(self, key, win):
        n = win["n"]
        if n < self.min_count:
            self._c["windows_below_min"] += 1
            return
        self._c["windows_scored"] += 1
        vxi, vyi, w_idx = key
        day = self._day_idx
        share = win["resp"] / n
        pbar = win["pi"] / n
        vbar = win["var"] / n
        bursty = (win["close_gaps"] / n) > self.burst_frac
        count_lbl = self._bin_label(n, self.count_edges, "n")
        ev_lbl = self._bin_label(win["ncross"], self.evidence_edges, "cr")
        cluster = self._clusters.setdefault((vxi, vyi, day), _CalTally(self.n_bins))
        voxel = self._voxels.setdefault((vxi, vyi), _CalTally(self.n_bins))

        for k in range(self._K):
            p = float(pbar[k])
            v = float(vbar[k])
            self._c["samples_total"] += 1
            if not (0.0 < p < 1.0):
                self._c["samples_degenerate"] += 1
                continue
            bernoulli = p * (1.0 - p)
            capped = v > self.cap_frac * bernoulli
            if capped:
                self._c["samples_capped"] += 1
                v = self.cap_frac * bernoulli
            clamped = bool(win["clamp"][k])
            if clamped:
                self._c["samples_clamped"] += 1
            order0 = int(win["order"][k]) == 0
            if order0:
                self._c["samples_order0"] += 1

            if v > 1.0e-12:
                nu = bernoulli / v - 1.0
                pmf, cdf = _beta_binomial_pmf_cdf(n, p * nu, (1.0 - p) * nu)
                pshare = (n - 1.0) / (nu + n)
            else:
                pmf, cdf = _binomial_pmf_cdf(n, p)
                pshare = 0.0

            u_soft = _stable_uniform(self.seed, "soft", vxi, vyi, day, w_idx, k)
            u_hard = _stable_uniform(self.seed, "hard", vxi, vyi, day, w_idx, k)
            # The soft share is a contracted count: each vote puts ~1-eps on
            # its hot slot and spreads eps over the rest, so n*share sits just
            # off the integer grid (in particular just ABOVE 0 on zero-count
            # slots). Scored at ceil, that off-grid shift dumps each count's
            # whole probability bin below the credible point and biases the
            # PIT upward on rare slots; mapping to the NEAREST count recovers
            # the exact count wherever responsibilities are peaked, and errs
            # by at most one (narrow) bin where contamination x count grows.
            x = float(np.floor(n * float(share[k]) + 0.5))
            cm_soft = _randomized_pit(pmf, cdf, x, u_soft)
            cm_hard = _randomized_pit(pmf, cdf, float(win["hard"][k]), u_hard)

            # Central 95% width of the soft predictive, in share units.
            q_lo = int(np.searchsorted(cdf, 0.025))
            q_hi = int(np.searchsorted(cdf, 0.975))
            width = (q_hi - q_lo) / n

            self._tallies["soft"].add(cm_soft, self.thresholds, width)
            self._tallies["hard"].add(cm_hard, self.thresholds, width)
            cluster.add(cm_soft, self.thresholds, width)
            voxel.add(cm_soft, self.thresholds, width)
            self._stratum("count", count_lbl).add(cm_soft, self.thresholds, width)
            self._stratum("evidence", ev_lbl).add(cm_soft, self.thresholds, width)
            self._stratum("slot", f"k{k}").add(cm_soft, self.thresholds, width)
            self._stratum("clamp", "clamped" if clamped else "clean").add(
                cm_soft, self.thresholds, width)
            self._stratum("cap", "capped" if capped else "free").add(
                cm_soft, self.thresholds, width)
            self._stratum("order", "order0" if order0 else "harmonic").add(
                cm_soft, self.thresholds, width)
            self._stratum("burst", "bursty" if bursty else "steady").add(
                cm_soft, self.thresholds, width)

            ps = self._pshare.setdefault(ev_lbl, [0.0, 0])
            if not capped:
                ps[0] += pshare
                ps[1] += 1

    # ---------------------------------------------------------------- result
    def _bootstrap(self):
        """Percentile CIs for the soft headline by resampling (voxel, day)
        clusters with replacement: the honest precision under within-cluster
        dependence (shared pedestrians, persistent per-voxel bias)."""
        clusters = list(self._clusters.values())
        if len(clusters) < 2:
            return None
        counts = np.stack([c.counts for c in clusters])
        cov68 = np.array([c.cov68 for c in clusters], float)
        cov95 = np.array([c.cov95 for c in clusters], float)
        ns = np.array([c.n for c in clusters], float)
        rng = np.random.default_rng(self.seed + 1)
        e, c68, c95 = [], [], []
        for _ in range(self.bootstrap_b):
            idx = rng.integers(0, len(clusters), len(clusters))
            nn = ns[idx].sum()
            if nn == 0:
                continue
            emp = counts[idx].sum(axis=0) / nn
            e.append(float(np.mean(np.abs(emp - self.thresholds))))
            c68.append(cov68[idx].sum() / nn)
            c95.append(cov95[idx].sum() / nn)

        def ci(a):
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

        return {"n_clusters": len(clusters), "b": self.bootstrap_b,
                "ece_ci": ci(e), "coverage_68_ci": ci(c68),
                "coverage_95_ci": ci(c95)}

    def _ece_null(self, n_samples):
        """Simulated null level of the finite-sample ECE: the value a PERFECT
        model shows on this many iid-uniform credible masses. The measured ECE
        must clear this, not zero."""
        if n_samples == 0:
            return None
        rng = np.random.default_rng(self.seed + 2)
        vals = []
        for _ in range(self.ece_null_m):
            u = rng.random(n_samples)
            emp = np.array([(u <= p).mean() for p in self.thresholds])
            vals.append(float(np.mean(np.abs(emp - self.thresholds))))
        return {"mean": float(np.mean(vals)),
                "p95": float(np.percentile(vals, 95.0)),
                "m": self.ece_null_m}

    def result(self):
        # Flush: end the last day (scores complete windows, drops truncated).
        self._end_day()

        out = {"dt_win_s": self.dt, "min_count": self.min_count,
               "n_bins": self.n_bins, "seed": self.seed,
               "var_cap_frac": self.cap_frac}
        for name, tal in self._tallies.items():
            out[name] = tal.result(self.thresholds)
        out["bootstrap"] = self._bootstrap()
        out["ece_null"] = self._ece_null(self._tallies["soft"].n)
        out["strata"] = {fam: {lbl: tal.result(self.thresholds)
                               for lbl, tal in sorted(d.items())}
                         for fam, d in self._strata.items()}
        # Voxel-balanced companion: unweighted mean over voxels of each
        # voxel's own coverage, so busy voxels cannot hide quiet-region
        # failures behind the pooled (traffic-weighted) headline.
        vox = [t for t in self._voxels.values() if t.n > 0]
        if vox:
            out["voxel_balanced"] = {
                "coverage_68": float(np.mean([t.cov68 / t.n for t in vox])),
                "coverage_95": float(np.mean([t.cov95 / t.n for t in vox])),
                "n_voxels": len(vox)}
        else:
            out["voxel_balanced"] = None
        out["parameter_share"] = {
            lbl: {"mean": (s / c if c else None), "n": c}
            for lbl, (s, c) in sorted(self._pshare.items())}
        total_votes = self._c["votes_matched"] + self._c["votes_unmatched"]
        out["matched_vote_fraction"] = (
            self._c["votes_matched"] / total_votes if total_votes else None)
        out["counters"] = dict(self._c)
        return out
