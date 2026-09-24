"""Ground-truth replay for the people-finding evaluation.

An episode starts at a node at a departure time with a deadline. The robot
moves on the traversable graph at a fixed speed along a planned waypoint
sequence (or stays put). The oracle replays the held-out detections and
reports every distinct person that comes within the encounter radius of the
robot, with the time of the first encounter.

Detections come from the per-day npz files (t [s], x/y [mm], id): the ATC day
files, or hb_to_npz.py for HB, row-aligned with the canonical stream.

Selftest: a synthetic day with one person crossing a known point at a known
time; a robot waiting there encounters them at the right time, a robot
elsewhere never does.

Usage:
    python3 encounter_oracle.py --help

Ground-truth replay: given an episode set and the recorded tracks, reports what
the planner would actually have encountered. This is the reference the
activity-table sources are scored against, so it takes no model.
"""
from __future__ import annotations

import numpy as np

ROBOT_SPEED_MPS = 1.5
STEP_S = 0.5          # robot-position sampling step for the replay


class DayDetections:
    """One day's detections, time-sorted, queryable by window."""

    def __init__(self, t, x_m, y_m, ids):
        order = np.argsort(t, kind="stable")
        self.t = np.asarray(t, dtype=np.float64)[order]
        self.x = np.asarray(x_m, dtype=np.float64)[order]
        self.y = np.asarray(y_m, dtype=np.float64)[order]
        self.id = np.asarray(ids, dtype=np.int64)[order]

    @classmethod
    def from_npz(cls, path):
        z = np.load(path)
        return cls(z["t"], z["x"] / 1000.0, z["y"] / 1000.0, z["id"])

    def window(self, t0, t1):
        lo = np.searchsorted(self.t, t0, side="left")
        hi = np.searchsorted(self.t, t1, side="left")
        return lo, hi


def robot_track(waypoints_xy, t_depart, speed=ROBOT_SPEED_MPS, step=STEP_S,
                t_end=None):
    """Sampled (t, x, y) track along a waypoint polyline at constant speed.
    A single waypoint means the robot waits there. If t_end is given, the
    track is extended (waiting at the final waypoint) or truncated to it."""
    wp = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_t = cum[-1] / speed
    end = total_t if t_end is None else max(t_end - t_depart, 0.0)
    ts = np.arange(0.0, end + step, step)
    s = np.minimum(ts * speed, cum[-1])
    xy = np.empty((len(ts), 2))
    if len(wp) == 1 or cum[-1] < 1e-9:
        xy[:] = wp[0]
    else:
        xy[:, 0] = np.interp(s, cum, wp[:, 0])
        xy[:, 1] = np.interp(s, cum, wp[:, 1])
    return ts + t_depart, xy


def replay(day: DayDetections, track_t, track_xy, radius_m=2.0):
    """Encounters along a sampled robot track.

    Returns (first_encounter_t or None, encounter ids in order of first
    contact). A detection counts when it falls within radius_m of the
    robot's position in the matching time step."""
    seen: dict = {}
    half = 0.5 * (track_t[1] - track_t[0]) if len(track_t) > 1 else 0.25
    for k in range(len(track_t)):
        lo, hi = day.window(track_t[k] - half, track_t[k] + half)
        if hi <= lo:
            continue
        dx = day.x[lo:hi] - track_xy[k, 0]
        dy = day.y[lo:hi] - track_xy[k, 1]
        close = (dx * dx + dy * dy) <= radius_m * radius_m
        for pid in day.id[lo:hi][close]:
            if pid not in seen:
                seen[int(pid)] = float(track_t[k])
    if not seen:
        return None, []
    ordered = sorted(seen.items(), key=lambda kv: kv[1])
    return ordered[0][1], [pid for pid, _ in ordered]


def selftest():
    # One person walks through (10, 0) at t=100 s, moving along +x at 1 m/s,
    # sampled at 10 Hz for 20 s (x from 0 to 20).
    ts = np.arange(90.0, 110.0, 0.1)
    xs = ts - 90.0
    day = DayDetections(ts, xs, np.zeros_like(ts), np.full(len(ts), 7))

    # Robot A waits at (10, 0): first encounter when the person reaches
    # within 2 m of x=10, i.e. x=8 at t=98.
    t_a, ids_a = replay(day, *robot_track([(10.0, 0.0)], 0.0, t_end=200.0))
    assert ids_a == [7], ids_a
    assert abs(t_a - 98.0) <= 0.5, t_a

    # Robot B waits far away: no encounter.
    t_b, ids_b = replay(day, *robot_track([(10.0, 50.0)], 0.0, t_end=200.0))
    assert t_b is None and ids_b == [], (t_b, ids_b)

    # Robot C drives from (0,0) to (20,0) starting at t=95 at 1.5 m/s and
    # chases the person (both move along +x): gap 5 m at t=95, closing
    # speed 0.5 m/s, radius 2 m -> caught after 6 s, at t=101.
    t_c, ids_c = replay(day, *robot_track([(0.0, 0.0), (20.0, 0.0)], 95.0))
    assert ids_c == [7], ids_c
    assert abs(t_c - 101.0) <= 0.6, t_c

    # Dedup: the person stays within radius of a waiting robot for ~4 s
    # (dozens of close samples) and still counts exactly once.
    t_d, ids_d = replay(day, *robot_track([(15.0, 0.0)], 90.0, t_end=120.0))
    assert ids_d == [7], ids_d
    assert abs(t_d - 103.0) <= 0.5, t_d
    print("encounter_oracle selftest OK")


if __name__ == "__main__":
    selftest()
