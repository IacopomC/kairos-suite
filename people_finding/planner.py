"""The fixed decision policy of the people-finding evaluation.

Finite-horizon dynamic programming over (traversable node, time step),
following the encounter-probability planner of Tipaldi & Arras (ICRA 2011,
Algorithm 1): actions are wait or move to an adjacent node, the reward of
occupying a node is the information source's predicted activity there at
that time, and the plan maximizes the accumulated reward within the
deadline. The planner is identical for every information source; only the
per-(cell, slot) activity table changes.

The time step equals one edge traversal (VOXEL / speed), so motion and
reward accrual share one clock. Activity tables are given in coarser slots
(e.g. 600 s) and are looked up per step.

Backward induction is vectorized over edges; the value table J (n_steps+1,
n_nodes) is kept and the action sequence is reconstructed forward, so no
per-node argmax arrays are stored.

Usage: imported, not run directly.

    from planner import VOXEL, load_graph

The fixed decision policy every activity source is evaluated under. It is
deliberately the same for all of them, so a difference in the result comes from
the activity table and not from the policy.
"""
from __future__ import annotations

import json

import numpy as np

VOXEL = 0.4
ROBOT_SPEED_MPS = 1.5
STEP_S = VOXEL / ROBOT_SPEED_MPS
PLACES_LAYER = 3


def load_graph(path):
    """Traversable nodes of the authored graph: positions (N,2), edge index
    arrays (src, dst; both directions), and the node-id list."""
    doc = json.load(open(path))
    nodes = [n for n in doc["nodes"] if int(n["layer"]) == PLACES_LAYER
             and float(n["attributes"].get("distance", 0.0)) > 0]
    ids = [int(n["id"]) for n in nodes]
    idx = {v: i for i, v in enumerate(ids)}
    pos = np.array([n["attributes"]["position"][:2] for n in nodes])
    src, dst = [], []
    for e in doc["edges"]:
        s, t = int(e["source"]), int(e["target"])
        if s in idx and t in idx:
            src += [idx[s], idx[t]]
            dst += [idx[t], idx[s]]
    return pos, ids, np.asarray(src, np.int64), np.asarray(dst, np.int64)


def plan(src, dst, n_nodes, reward_of, start, n_steps):
    """Backward induction over (node, step). reward_of(k) -> (N,) activity
    of occupying each node during step k. Returns node-index sequence of
    length n_steps+1 starting at `start`."""
    J = np.zeros((n_steps + 1, n_nodes))
    for k in range(n_steps - 1, -1, -1):
        Jn = J[k + 1] + reward_of(k + 1)
        Jb = Jn.copy()                        # stay
        np.maximum.at(Jb, src, Jn[dst])       # or move to a neighbor
        J[k] = Jb
    # forward reconstruction: at each step pick the neighbor (or self)
    # achieving the stored value
    nbrs = [[] for _ in range(n_nodes)]
    for a, b in zip(src, dst):
        nbrs[a].append(b)
    seq = [start]
    v = start
    for k in range(n_steps):
        Jn = J[k + 1] + reward_of(k + 1)
        best_v, best_j = v, Jn[v]
        for u in nbrs[v]:
            if Jn[u] > best_j:
                best_j = Jn[u]
                best_v = u
        seq.append(int(best_v))
        v = best_v
    return seq


def waypoints(seq, pos):
    """Collapse a node sequence to polyline waypoints (drop repeats)."""
    wp = [pos[seq[0]]]
    for v in seq[1:]:
        if not np.allclose(pos[v], wp[-1]):
            wp.append(pos[v])
    return np.asarray(wp)


def selftest():
    # A 1x5 line graph. A hotspot appears at node 4 only during steps 6..10;
    # node 0 has weak constant activity. Starting at node 0 with a deadline
    # of 12 steps, the plan must walk to node 4 (4 steps away) and be there
    # during the hotspot window.
    src = np.array([0, 1, 1, 2, 2, 3, 3, 4])
    dst = np.array([1, 0, 2, 1, 3, 2, 4, 3])

    def reward_of(k):
        r = np.zeros(5)
        r[0] = 0.05
        if 6 <= k <= 10:
            r[4] = 1.0
        return r

    seq = plan(src, dst, 5, reward_of, start=0, n_steps=12)
    assert seq[0] == 0
    assert all(seq[k] == 4 for k in range(6, 11)), seq
    seq2 = plan(src, dst, 5, lambda k: reward_of(0), start=0, n_steps=12)
    assert all(v == 0 for v in seq2), seq2

    # timing sanity at HB scale: ~4k nodes, ~16k directed edges, 900 steps
    rng = np.random.default_rng(0)
    n = 4000
    s2 = rng.integers(0, n, 16000)
    d2 = rng.integers(0, n, 16000)
    r = rng.random(n)
    import time
    tic = time.time()
    plan(np.r_[s2, d2], np.r_[d2, s2], n, lambda k: r, start=0, n_steps=900)
    print(f"planner selftest OK ({time.time()-tic:.2f}s for a 900-step DP "
          f"at HB scale)")


if __name__ == "__main__":
    selftest()
