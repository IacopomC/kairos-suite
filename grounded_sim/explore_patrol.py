"""Region patrols for the sparse-coverage study.

The environment is partitioned into k regions; at every decision interval the
robot picks a region uniformly at random (seeded) and observes it. Unlike the
reference protocol, the robot never teleports: it drives to the chosen region
on the traversable graph and covers it with a serpentine loop until the next
decision.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from robot_sim import ROBOT_SPEED_MPS, build_coverage_loop

DECIDE_DT_S = 600.0


def make_regions(pos, trav, k=24, seed=7):
    """k-means over traversable node positions -> region label per node
    (-1 for non-traversable)."""
    rng = np.random.default_rng(seed)
    tpos = pos[trav]
    centers = tpos[rng.choice(len(tpos), k, replace=False)]
    for _ in range(50):
        d = ((tpos[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.array([tpos[lab == j].mean(0) if (lab == j).any()
                        else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    labels = np.full(len(pos), -1, dtype=int)
    labels[np.flatnonzero(trav)] = lab
    return labels, centers


class RegionPatrols:
    """Per-region serpentine loops and inter-region travel paths on the
    traversable graph, precomputed once."""

    def __init__(self, pos, adj, trav, labels, k):
        self.pos = pos
        self.k = k
        self.loops = []
        self.entry = []          # a node index per region (loop start)
        g_idx = np.flatnonzero(trav)
        for j in range(k):
            mask = labels == j
            if mask.sum() < 4:
                idx = np.flatnonzero(mask)
                self.loops.append(pos[idx] if len(idx) else pos[g_idx[:1]])
                self.entry.append(idx[0] if len(idx) else g_idx[0])
                continue
            loop, order = build_coverage_loop(pos, adj, mask, lane_width=4.0)
            self.loops.append(np.asarray(loop)[:, :2])
            self.entry.append(int(order[0]) if len(order) else
                              int(np.flatnonzero(mask)[0]))
        # all-pairs shortest paths between region entries (predecessors for
        # path extraction)
        self.adj = adj
        entries = np.asarray(self.entry)
        self.dist, self.pred = dijkstra(adj, indices=entries,
                                        return_predecessors=True)
        self.entries = entries

    def travel(self, j_from, j_to):
        """Node path from region j_from's entry to j_to's entry."""
        tgt = self.entries[j_to]
        path = [int(tgt)]
        while path[-1] != self.entries[j_from]:
            p = self.pred[j_from, path[-1]]
            if p < 0:
                break
            path.append(int(p))
        return self.pos[np.asarray(path[::-1])][:, :2]


def day_polyline(patrols, schedule, day_dur_s, speed=ROBOT_SPEED_MPS,
                 decide_dt_s=DECIDE_DT_S):
    """Travel + in-region coverage for one day's region schedule, one
    region per decision interval, as a single polyline the RobotSim drives
    at constant speed. Each interval consumes decide_dt_s of arclength
    (travel to the region included), so the schedule's timing is honored."""
    pts = [patrols.loops[schedule[0]][0]]
    cur = schedule[0]
    for j in schedule:
        target = decide_dt_s * speed
        used = 0.0
        if j != cur:
            leg = patrols.travel(cur, j)
            used += _polylen(np.vstack([pts[-1][None, :], leg]))
            pts.extend(leg)
            cur = j
        loop = np.asarray(patrols.loops[cur])
        while used < target:
            for a, b in zip(loop[:-1], loop[1:]):
                seg = float(np.linalg.norm(b - a))
                pts.append(b)
                used += seg
                if used >= target:
                    break
            if len(loop) < 2:
                break
    return np.asarray(pts)


def _polylen(p):
    p = np.asarray(p)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def sample_schedule(weights, n_intervals, rng):
    """Region index per interval, drawn independently with probability
    proportional to `weights` (uniform if degenerate)."""
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return rng.choice(len(w), size=n_intervals, p=w)


def active_intervals(n_iv, ratio):
    """Boolean mask over a day's `n_iv` decision intervals marking the ones
    the robot actually observes, implementing the exploration ratio of the
    sparse-coverage protocol: only n_active =
    round(ratio * n_iv) of the intervals gather data, the rest the robot is
    occupied elsewhere. The active intervals are spread across the day by even
    stride (their round-robin-in-time schedule), so lowering the ratio thins
    observation without biasing the time of day it happens. ratio >= 1 marks
    every interval."""
    n_iv = int(n_iv)
    if ratio >= 1.0:
        return np.ones(n_iv, dtype=bool)
    n_active = max(1, int(round(ratio * n_iv)))
    idx = np.unique(np.linspace(0, n_iv - 1, n_active).round().astype(int))
    mask = np.zeros(n_iv, dtype=bool)
    mask[idx] = True
    return mask
