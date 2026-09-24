#!/usr/bin/env python3
r"""Per-cell statistics for the sparse-embodied-coverage tables.

Reads the ``cellmap.npz`` dumps written by ``evaluate.py`` (per-cell circular
CRPS, heading MLPD, and MPP keyed to voxel indices) for the uniformly random
region patrol, averages over seeds, and reports the numbers behind the
sparse-coverage table: per-cell CRPS (with a bootstrap 95% CI over cells) and
heading MLPD for each method, plus the paired per-cell comparison against each
baseline.

Deterministic: the bootstrap uses a fixed seed so the CI is reproducible.

Example:
    python3 compute_sparse_coverage_stats.py --results-dir ../results/exploration
"""
import argparse
import numpy as np

PREFIX = {"kairos": "k"}
# Comparison methods are added at the command line as NAME:DIR_PREFIX pairs,
# since their runs are produced outside this suite.
POLICY = "random"


def load_run(results_dir, method, policy, seeds):
    """Seed-averaged per-cell metrics; returns {(vx, vy): (crps, mlpd, mpp)}."""
    per = []
    for s in seeds:
        d = np.load(f"{results_dir}/{PREFIX[method]}_{policy}_s{s}/cellmap.npz")
        per.append({(int(a), int(b)): (float(c), float(m), float(p))
                    for a, b, c, m, p in
                    zip(d["vx"], d["vy"], d["crps"], d["mlpd"], d["mpp"])})
    keys = set(per[0])
    for m in per[1:]:
        keys &= set(m)
    return {k: tuple(np.mean([mm[k][i] for mm in per]) for i in range(3))
            for k in keys}


def col(run, i):
    return np.array([v[i] for v in run.values()])


def bootstrap_ci(x, rng, B=2000):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = x[idx].mean(axis=1)
    return np.percentile(bs, [2.5, 97.5])


def paired(A, B):
    """A minus B on shared cells: (mean diff, median diff, %% of cells A<B)."""
    ks = sorted(set(A) & set(B))
    d = np.array([A[k][0] - B[k][0] for k in ks])
    return d.mean(), np.median(d), (d < 0).mean() * 100, len(ks)


def report(results_dir, seeds, label):
    rng = np.random.default_rng(0)
    print(f"\n===== {label}  ({results_dir}) =====")
    runs = {}
    for method in PREFIX:
        try:
            runs[method] = load_run(results_dir, method, POLICY, seeds)
        except FileNotFoundError:
            pass
    print(f"{'method':9s} {'CRPS':>7s} {'[95% CI]':>18s} "
          f"{'H-MLPD':>9s} {'MPP':>7s} {'ncell':>7s}")
    for method, run in runs.items():
        crps = col(run, 0); lo, hi = bootstrap_ci(crps, rng)
        print(f"{method:9s} {crps.mean():7.4f} "
              f"[{lo:6.4f},{hi:6.4f}] {col(run,1).mean():+9.4f} "
              f"{col(run,2).mean():7.4f} {len(run):7d}")

    others = [m for m in PREFIX if m != "kairos"]
    if others:
        print("\n  -- per-cell comparison (Kairos vs other) --")
    for other in others:
        if "kairos" in runs and other in runs:
            md, mdn, win, n = paired(runs["kairos"], runs[other])
            print(f"    kairos-{other}: meanD={md:+.4f} "
                  f"medD={mdn:+.4f} Kairos-better={win:.0f}% of {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="../results/exploration",
                    help="84-day / 6.25%% run (~14 visits/region)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--compare", nargs="*", default=[], metavar="NAME:PREFIX",
                    help="comparison methods to read alongside Kairos, as "
                         "NAME:DIR_PREFIX (their runs come from outside this suite)")
    args = ap.parse_args()
    for spec in args.compare:
        name, _, prefix = spec.partition(":")
        PREFIX[name] = prefix or name

    report(args.results_dir, args.seeds, "84 days, 6.25% (~14 visits/region)")


if __name__ == "__main__":
    main()
