"""Pool the per-scene flow-timing stats into the numbers for the cost table.

Each scene's train/test.stats.json carries, per timer, {count, total_us,
median_ms, p95_ms, std_ms} (std_ms added to the pybind flow_timing export).
Because every event is recorded, the per-scene mean equals total_us/count, and
we pool across scenes exactly:

    mean  = sum(total_us) / sum(count)                       (exact)
    std   = sqrt( sum(n_i*(sd_i^2 + mu_i^2))/N - mu^2 )      (exact pooled pop. std)
    med/p95 = count-weighted mean of the per-scene percentiles (approximation;
              exact pooling would need the raw samples, which are not dumped)

Usage:
    python aggregate_flow_timing.py <results_dir>     # e.g. results/kairos_timing_std
"""

import glob
import json
import math
import os
import sys

# table row -> flow_timing key prefix
ROWS = [
    ("Per frame (10\\,Hz)",            "process_frame"),
    ("Cell update (per det.)",         "add_observation"),
    ("Sharing (per neighbourhood)",    "sharing"),
    ("Coupling (per frame it runs)",   "coupling"),
    ("Node summary (per event)",       "place_summaries"),
    ("Deformation remap (per event)",  "remap_cells"),
]


def collect(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*", "*", "*.stats.json")))
    scenes = []
    for f in files:
        try:
            ft = json.load(open(f)).get("flow_timing")
        except Exception:
            continue
        if ft:
            scenes.append((f, ft))
    return scenes


def pool(scenes, prefix):
    N = 0
    tot_us = 0.0
    ex2_acc = 0.0          # sum n_i * (sd_i^2 + mu_i^2)   [ms^2]
    med_acc = p95_acc = 0.0
    have_std = True
    for _, ft in scenes:
        n = ft.get(f"{prefix}_count") or 0
        t = ft.get(f"{prefix}_total_us")
        if not n or t is None:
            continue
        mu = (t / n) / 1000.0                       # ms
        sd = ft.get(f"{prefix}_std_ms")
        if sd is None:
            have_std = False
            sd = 0.0
        N += n
        tot_us += t
        ex2_acc += n * (sd * sd + mu * mu)
        med_acc += n * (ft.get(f"{prefix}_median_ms") or 0.0)
        p95_acc += n * (ft.get(f"{prefix}_p95_ms") or 0.0)
    if N == 0:
        return None
    mean = (tot_us / N) / 1000.0
    var = max(0.0, ex2_acc / N - mean * mean)
    std = math.sqrt(var) if have_std else float("nan")
    return dict(n=N, mean=mean, std=std, median=med_acc / N, p95=p95_acc / N,
                have_std=have_std)


def fmt(x, d=3):
    return f"{x:.{d}f}" if x == x else "nan"  # nan check


def main(results_dir):
    scenes = collect(results_dir)
    print(f"# {len(scenes)} scenes with flow_timing under {results_dir}\n")
    print(f"{'row':30s} {'n':>10s} {'mean':>9s} {'std':>9s} {'median':>9s} {'p95':>9s}")
    latex = []
    for label, prefix in ROWS:
        r = pool(scenes, prefix)
        if r is None:
            print(f"{label:30s} {'--- no data ---':>10s}")
            continue
        warn = "" if r["have_std"] else "  (no std_ms in json -- rerun needed)"
        print(f"{label:30s} {r['n']:>10d} {fmt(r['mean'])!s:>9} {fmt(r['std'])!s:>9} "
              f"{fmt(r['median'])!s:>9} {fmt(r['p95'])!s:>9}{warn}")
        latex.append(f"\\quad {label} & ${fmt(r['mean'])}\\pm{fmt(r['std'])}$ "
                     f"& ${fmt(r['median'])}$ & ${fmt(r['p95'])}$ \\\\")
    print("\n% --- LaTeX rows (mean$\\pm$std, median, p95) ---")
    print("\n".join(latex))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
