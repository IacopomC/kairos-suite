"""Baselines of the sparse-coverage study: STeF-Map or Aion is trained on
what a region-patrolling robot sees.

Same protocol as the Kairos region patrol (explore_patrol): 24 regions, one
uniformly random region per 600 s interval, travel paid on the traversable
graph, 5 m field of view, 84 training days. The masked detection stream
(within the robot's field of view) is fed to the system chronologically,
exactly as the Kairos trainer feeds its pipeline.

  stef (host):      python3 explore_baselines.py --system stef ...
  aion (its image): python3 explore_baselines.py --system aion ...

Output: the trained model (stef pickle / aion state) for the unified
scoring pass.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import os
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, os.environ.get("STEFMAP_DIR", "stefmap"))
sys.path.insert(0, "/ws/stefmap")

from atc_stream import AtcStream  # noqa: E402
from draw_patrol import load_nav_graph_json  # noqa: E402
from explore_patrol import (DECIDE_DT_S, RegionPatrols, active_intervals,  # noqa: E402
                            day_polyline, make_regions, sample_schedule)
from robot_sim import KEYFRAME_DT_S, ROBOT_SPEED_MPS, RobotSim  # noqa: E402

PERIODS_S = [3600.0, 43200.0, 86400.0, 604800.0]
FOV_M = 5.0


class StefAdapter:
    def __init__(self):
        from stefmap.flow_map import FlowMap
        self.m = FlowMap(cell_size=0.4, n_bins=8, time_window_s=10.0,
                         fremen_order=1, candidate_periods=PERIODS_S)
        self._window_end = None
        self._tw_ns = int(10.0 * 1e9)

    def feed(self, t_s, x, y, theta):
        for i in range(len(t_s)):
            ti = int(t_s[i] * 1e9)
            if self._window_end is None:
                self._window_end = ti + self._tw_ns
            while ti >= self._window_end:
                self.m.flush_window(self._window_end)
                self._window_end += self._tw_ns
            self.m.add_detection(ti, float(x[i]), float(y[i]), float(theta[i]))

    def save(self, path):
        self.m._window_counts = dict(self.m._window_counts)
        with open(path, "wb") as f:
            pickle.dump(self.m, f)


class AionAdapter:
    def __init__(self):
        import aion_python
        self.m = aion_python.AionCore()
        self._window_end = None
        self._tw_ns = int(10.0 * 1e9)

    def feed(self, t_s, x, y, theta):
        for i in range(len(t_s)):
            ti = int(t_s[i] * 1e9)
            if self._window_end is None:
                self._window_end = ti + self._tw_ns
            while ti >= self._window_end:
                self.m.flush_window(self._window_end)
                self._window_end += self._tw_ns
            self.m.add_detection(ti, float(x[i]), float(y[i]), float(theta[i]))

    def save(self, path):
        ok, err = self.m.save_state(path)
        if not ok:
            raise SystemExit(f"aion save_state failed: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["stef", "aion"])
    ap.add_argument("--policy", default="random", choices=["random"],
                    help="region choice per interval (uniformly random)")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--days", nargs="+", required=True, help="train-day npz")
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--policy-seed", type=int, default=7)
    ap.add_argument("--policy-regions", type=int, default=24)
    ap.add_argument("--explore-ratio", type=float, default=1.0,
                    help="fraction of the day's 10-min decision intervals the "
                         "robot observes (Molina et al.'s exploration ratio); "
                         "active intervals spread by even stride; 1.0 = all day")
    ap.add_argument("--robot-speed", type=float, default=ROBOT_SPEED_MPS,
                    help="robot travel speed (m/s)")
    ap.add_argument("--fov-m", type=float, default=FOV_M,
                    help="field-of-view radius for the robot")
    args = ap.parse_args()

    pos, ids, adj, trav = load_nav_graph_json(args.graph)
    labels, _ = make_regions(pos, trav, k=args.policy_regions,
                             seed=args.policy_seed)
    patrols = RegionPatrols(pos, adj, trav, labels, args.policy_regions)
    rng = np.random.default_rng(args.policy_seed)
    xb = (pos[:, 0].min() - 6, pos[:, 0].max() + 6)
    yb = (pos[:, 1].min() - 6, pos[:, 1].max() + 6)

    model = StefAdapter() if args.system == "stef" else AionAdapter()

    total = 0
    tic = time.time()
    for di, day in enumerate(args.days):
        s = AtcStream(day, x_bounds=xb, y_bounds=yb)
        dur = s.t_max - s.t_min
        n_iv = max(1, int(np.ceil(dur / DECIDE_DT_S)))
        w = np.ones(args.policy_regions)
        schedule = sample_schedule(w, n_iv, rng)
        sim = RobotSim(day_polyline(patrols, schedule, dur,
                                    speed=args.robot_speed), pos,
                       radius=args.fov_m)
        active = active_intervals(n_iv, args.explore_ratio)

        # mask the day's stream by the robot's field of view, vectorized:
        # keyframe index per detection -> robot pose -> distance gate
        kf = np.floor((s.t - s.t_min) / KEYFRAME_DT_S).astype(np.int64)
        arc = (args.robot_speed * KEYFRAME_DT_S) * kf
        # pose_at_arclength is scalar; interpolate poses on the polyline once
        seg = np.linalg.norm(np.diff(sim.poly[:, :2], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        sarc = np.mod(arc, cum[-1])
        px = np.interp(sarc, cum, sim.poly[:, 0])
        py = np.interp(sarc, cum, sim.poly[:, 1])
        # exploration ratio: keep detections only in the observed intervals
        iv = np.clip(np.floor((s.t - s.t_min) / DECIDE_DT_S).astype(np.int64),
                     0, n_iv - 1)
        keep = (((s.x - px) ** 2 + (s.y - py) ** 2) <= args.fov_m ** 2) & active[iv]
        model.feed(s.t[keep], s.x[keep], s.y[keep], s.theta[keep])
        total += int(keep.sum())
        print(f"[explore-{args.system}-{args.policy}] "
              f"{pathlib.Path(day).stem}: fed {int(keep.sum())} of "
              f"{len(s.t)} ({(time.time()-tic)/60:.1f} min)", flush=True)
    model.save(args.out_model)
    print(f"[explore-{args.system}-{args.policy}] {total} detections total; "
          f"model -> {args.out_model}")


if __name__ == "__main__":
    main()
