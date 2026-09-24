"""Confidence intervals for the encounter-probability planning results.

The informed planners are deterministic over a frozen state, so there is no
seed variance to average; the genuine uncertainty is generalization across
held-out days. Because every day repeats the same fixed grid of departure
times from a single fixed home node, the held-out DAY is the unit of
replication. We therefore report each success rate (and, optionally, the mean
encounters) as a 95% t-interval over the D held-out days.

Point estimates equal the pooled values reported in the table because the
per-day episode counts are balanced.

Usage:
    python compute_finding_ci.py <finding.json> [more.json ...]
"""

import json
import sys
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy import stats  # only for the t critical value

SOURCES = ["wait", "random", "static", "fremen", "kairos"]


def day_summaries(records, source):
    """Return per-day (success_rate, mean_enc) arrays for one source."""
    by_day = defaultdict(list)
    for r in records:
        if r["source"] == source:
            by_day[r["day"]].append(r)
    days = sorted(by_day)
    succ = np.array([np.mean([e["t_find"] is not None for e in by_day[d]]) for d in days])
    enc = np.array([np.mean([e["n_enc"] for e in by_day[d]]) for d in days])
    return days, succ, enc


def t_interval(per_day, level=0.95):
    """95% t-interval over the D held-out days (D-1 dof)."""
    d = len(per_day)
    mean = float(per_day.mean())
    se = float(per_day.std(ddof=1) / sqrt(d))
    tc = float(stats.t.ppf(0.5 + level / 2.0, d - 1))
    return mean, mean - tc * se, mean + tc * se


def main(paths):
    for path in paths:
        blob = json.load(open(path))
        records, summary = blob["records"], blob["summary"]
        days = sorted({r["day"] for r in records})
        print(f"\n=== {path}  (D={len(days)} held-out days) ===")
        print(f"{'source':9s} {'succ':>6s} {'succ 95% CI':>18s}   {'enc':>8s} {'enc 95% CI':>18s}")
        for a in SOURCES:
            _, succ, enc = day_summaries(records, a)
            sm, slo, shi = t_interval(succ)
            em, elo, ehi = t_interval(enc)
            # guard: pooled point estimate must match the stored summary
            assert abs(sm - summary[a]["success"]) < 1e-9, (a, sm, summary[a]["success"])
            slo, shi = max(0.0, slo), min(1.0, shi)
            print(f"{a:9s} {sm:6.3f} [{slo:6.3f}, {shi:6.3f}]   "
                  f"{em:8.2f} [{max(0.0, elo):6.2f}, {ehi:6.2f}]")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    main(args)
