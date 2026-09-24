r"""Run the people-finding evaluation: one fixed planner, one episode set,
the information source swapped per source.

Episodes: departures every --depart-step-s within --hours (local) on every
held-out day; the robot starts at the node nearest the patrol loop's start
pose and has --deadline-s to meet people. Every source is scored by the same
ground-truth replay (encounter = distinct person within --radius-m).

Sources: 'wait' and 'random' are the uninformed floors; every A_<name>.npz in
the tables directory becomes an informed source running the identical DP.

    python3 run_finding.py --graph <hb_flow_graph.json> \
        --npz-dir <per-day npz from hb_to_npz.py> \
        --tables ../results/people_finding/hb \
        --out ../results/people_finding/hb/finding.json
"""
from __future__ import annotations

import argparse
import os
import json
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from encounter_oracle import DayDetections, replay, robot_track  # noqa: E402
from planner import STEP_S, load_graph, plan, waypoints  # noqa: E402

SLOT_S = 600.0
N_SLOTS = int(86400 / SLOT_S)


def episode_departures(day0, hours, step_s, tz_s):
    t = np.arange(day0, day0 + 86400.0, step_s)
    hh = ((t + tz_s) % 86400.0) / 3600.0       # local hours
    return t[(hh >= hours[0]) & (hh < hours[1])]


def day0_epoch(day_iso):
    from datetime import date
    s = day_iso.replace("-", "")
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    return (date(y, m, d) - date(1970, 1, 1)).days * 86400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph",
                    default=os.path.join(
                        os.environ.get("DATA_ROOT",
                                       os.path.expanduser("~/datasets")),
                        "hb-station", "graph", "hb_flow_graph.json"))
    ap.add_argument("--npz-dir", required=True,
                    help="per-day detection npz (t, x/y mm, id)")
    ap.add_argument("--tables", required=True,
                    help="directory with A_<source>.npz activity tables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-s", type=float, default=600.0)
    ap.add_argument("--depart-step-s", type=float, default=1800.0)
    ap.add_argument("--hours", type=float, nargs=2, default=(5.0, 22.0))
    ap.add_argument("--tz-s", type=float, default=7200.0,
                    help="local-time offset for --hours (HB 7200, ATC 32400)")
    ap.add_argument("--radius-m", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pos, ids, src, dst = load_graph(args.graph)
    n = len(ids)
    nbrs = [[] for _ in range(n)]
    for a, b in zip(src, dst):
        nbrs[a].append(b)

    # start node: nearest to the patrol loop's start (the docking pose)
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from draw_patrol import load_nav_graph_json
    from robot_sim import build_coverage_loop
    gpos, gids, gadj, gtrav = load_nav_graph_json(args.graph)
    poly, _ = build_coverage_loop(gpos, gadj, gtrav)
    start = int(np.argmin(np.linalg.norm(pos - np.asarray(poly[0][:2]), axis=1)))
    print(f"[finding] start node {start} at {pos[start]}")

    tables = {}
    for f in sorted(pathlib.Path(args.tables).glob("A_*.npz")):
        z = np.load(f, allow_pickle=True)
        name = f.stem[2:]
        tables[name] = dict(A=z["A"], days=[str(d) for d in z["days"]])
        assert z["A"].shape[2] == n, (name, z["A"].shape)
        print(f"[finding] table {name}: {z['A'].shape}")

    n_steps = int(round(args.deadline_s / STEP_S))
    rng = np.random.default_rng(args.seed)
    day_files = sorted(pathlib.Path(args.npz_dir).glob("*.npz"))
    records = []
    tic = time.time()
    for f in day_files:
        day_iso = f.stem.split("-", 1)[1]
        day = DayDetections.from_npz(f)
        d0 = day0_epoch(day_iso)
        departs = episode_departures(d0, args.hours, args.depart_step_s,
                                     args.tz_s)
        for t_dep in departs:
            for name in ["wait", "random"] + sorted(tables):
                if name == "wait":
                    wp = pos[[start]]
                elif name == "random":
                    seq = [start]
                    for _ in range(n_steps):
                        cand = nbrs[seq[-1]]
                        seq.append(int(rng.choice(cand)) if cand else seq[-1])
                    wp = waypoints(seq, pos)
                else:
                    tab = tables[name]
                    di = tab["days"].index(day_iso)
                    A = tab["A"][di]

                    def reward_of(k, A=A, t_dep=t_dep):
                        t = t_dep + k * STEP_S
                        si = min(int((t - d0) / SLOT_S), N_SLOTS - 1)
                        return A[si]

                    seq = plan(src, dst, n, reward_of, start, n_steps)
                    wp = waypoints(seq, pos)
                tt, xy = robot_track(wp, t_dep, t_end=t_dep + args.deadline_s)
                t_first, enc = replay(day, tt, xy, radius_m=args.radius_m)
                records.append(dict(day=day_iso, t=float(t_dep), source=name,
                                    t_find=(None if t_first is None
                                            else float(t_first - t_dep)),
                                    n_enc=len(enc)))
        print(f"  [finding] {day_iso}: {len(departs)} departures done "
              f"({(time.time()-tic)/60:.1f} min)", flush=True)

    sources = sorted({r["source"] for r in records})
    print("\nsource         success   median-t-find[s]  mean-encounters  (n)")
    summary = {}
    for a in sources:
        rs = [r for r in records if r["source"] == a]
        found = [r["t_find"] for r in rs if r["t_find"] is not None]
        s = dict(success=len(found) / len(rs),
                 median_t=float(np.median(found)) if found else None,
                 mean_enc=float(np.mean([r["n_enc"] for r in rs])),
                 n=len(rs))
        summary[a] = s
        print(f"{a:14s} {s['success']:8.3f}   "
              f"{s['median_t'] if s['median_t'] is not None else float('nan'):16.1f}  "
              f"{s['mean_enc']:15.3f}  ({s['n']})")
    with open(args.out, "w") as fh:
        json.dump(dict(records=records, summary=summary,
                       config=vars(args)), fh)
    print(f"[finding] wrote {args.out}")


if __name__ == "__main__":
    main()
