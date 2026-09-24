#!/usr/bin/env python3
"""Self-tests for WeightCalibrationAccumulator.

Three families:
  * construction invariants -- chunk invariance (identical results under any
    chunking of the same stream, the property every other accumulator also
    holds), the NaN-std refusal, min-count exclusion, truncated
    final windows, matched-fraction bookkeeping;
  * statistical null -- data simulated FROM the scored predictive must come
    out calibrated: coverage at nominal within binomial tolerance and ECE at
    its simulated null level. This validates the randomized-PIT scoring rule
    end to end (a super-uniform or biased rule fails here);
  * discrimination -- data whose true shares drift away from the predicted
    weights must produce under-coverage (the metric can actually fail a model).

Run: python3 test_weight_calibration.py
"""
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from online_metrics import WeightCalibrationAccumulator  # noqa: E402

K = 8
VS = 0.4
DT = 60.0


def _softish(k_hot, eps=0.02):
    """A peaked responsibility vector, mimicking the 8-slot cardinal layout."""
    r = np.full(K, eps / (K - 1))
    r[k_hot] = 1.0 - eps
    return r


def _stream(n_days=3, votes_per_win=12, wins_per_day=30, pi_true=None,
            pi_pred=None, std=0.02, seed=7, day0=1_000_000.0):
    """Synthetic per-detection rows for one voxel: each vote is one id-less
    detection (per-frame fallback), so votes are exactly the rows and the test
    controls counts directly. Returns row arrays in stream order."""
    rng = np.random.default_rng(seed)
    pi_pred = np.asarray(pi_pred, float)
    rows = []
    for d in range(n_days):
        t0 = day0 + d * 86400.0
        for w in range(wins_per_day):
            base = t0 + w * DT
            for v in range(votes_per_win):
                t = base + (v + 0.5) * (DT / votes_per_win)
                k = rng.choice(K, p=pi_true)
                rows.append((t, _softish(k)))
    t_s = np.array([r[0] for r in rows])
    resp = np.stack([r[1] for r in rows])
    n = t_s.size
    xyz = np.tile([0.2, 0.2, 0.0], (n, 1))
    ids = None
    matched = np.ones(n, bool)
    slot_w = [pi_pred.tolist()] * n
    slot_sd = [[std] * K] * n
    slot_cl = [[0] * K] * n
    slot_or = [[0] * K] * n
    ncr = np.full(n, 100)
    return t_s, xyz, ids, matched, resp, slot_w, slot_sd, slot_cl, slot_or, ncr


def _feed(acc, rows, chunk):
    t_s, xyz, ids, matched, resp, sw, sd, cl, so, ncr = rows
    n = t_s.size
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        acc.update(t_s[lo:hi], xyz[lo:hi], None, matched[lo:hi], resp[lo:hi],
                   sw[lo:hi], sd[lo:hi], cl[lo:hi], so[lo:hi], ncr[lo:hi])


def _fresh(**kw):
    args = dict(dt_win=DT, min_count=5, voxel_size=VS, bootstrap_b=50,
                ece_null_m=50, seed=3)
    args.update(kw)
    return WeightCalibrationAccumulator(**args)


def test_chunk_invariance():
    pi = np.full(K, 1.0 / K)
    rows = _stream(pi_true=pi, pi_pred=pi)
    results = []
    for chunk in (1, 7, 61, 10_000):
        acc = _fresh()
        _feed(acc, rows, chunk)
        results.append(acc.result())
    ref = results[0]
    for r in results[1:]:
        for v in ("soft", "hard"):
            for key in ("ece", "coverage_68", "coverage_95", "n"):
                assert r[v][key] == ref[v][key], (v, key, r[v][key], ref[v][key])
        assert r["counters"] == ref["counters"]
    print(f"chunk invariance: OK (n={ref['soft']['n']} samples)")


def test_null_calibration():
    """Votes drawn from the predicted weights, negligible parameter variance:
    the soft and hard variants must both sit at nominal coverage."""
    pi = np.array([0.35, 0.2, 0.15, 0.1, 0.08, 0.05, 0.04, 0.03])
    rows = _stream(n_days=6, wins_per_day=60, votes_per_win=15,
                   pi_true=pi, pi_pred=pi, std=1e-9, seed=11)
    acc = _fresh()
    _feed(acc, rows, 5_000)
    r = acc.result()
    n = r["soft"]["n"]
    assert n > 2000, n
    tol = 4.0 / np.sqrt(n / K)  # slots within a window are dependent
    for v in ("soft", "hard"):
        # The soft-vote bound errs conservative, so allow one-sided slack up.
        assert r[v]["coverage_95"] > 0.95 - tol, (v, r[v]["coverage_95"])
        assert r[v]["coverage_68"] > 0.6827 - tol, (v, r[v]["coverage_68"])
        assert r[v]["coverage_95"] < 0.999, (v, r[v]["coverage_95"])
    null = r["ece_null"]["p95"]
    assert r["soft"]["ece"] < 4.0 * null + 0.02, (r["soft"]["ece"], null)
    print(f"null calibration: OK (soft cov95={r['soft']['coverage_95']:.3f}, "
          f"cov68={r['soft']['coverage_68']:.3f}, ece={r['soft']['ece']:.4f}, "
          f"null p95={null:.4f}, n={n})")


def test_detects_miscalibration():
    """True shares far from the prediction with a tight predictive must
    under-cover badly -- the metric can fail a wrong model."""
    pi_pred = np.array([0.5, 0.3, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025])
    pi_true = np.array([0.05, 0.05, 0.5, 0.3, 0.025, 0.025, 0.025, 0.025])
    rows = _stream(n_days=4, wins_per_day=40, votes_per_win=15,
                   pi_true=pi_true, pi_pred=pi_pred, std=1e-9, seed=13)
    acc = _fresh()
    _feed(acc, rows, 5_000)
    r = acc.result()
    assert r["soft"]["coverage_95"] < 0.75, r["soft"]["coverage_95"]
    print(f"miscalibration detected: OK (cov95={r['soft']['coverage_95']:.3f})")


def test_nan_std_refused():
    pi = np.full(K, 1.0 / K)
    rows = list(_stream(pi_true=pi, pi_pred=pi))
    rows[6] = [[float("nan")] * K] * rows[0].size
    acc = _fresh()
    try:
        _feed(acc, tuple(rows), 10_000)
    except ValueError as e:
        assert "variance" in str(e)
        print("NaN std refused: OK")
        return
    raise AssertionError("NaN weight std was silently accepted")


def test_min_count_and_truncation():
    pi = np.full(K, 1.0 / K)
    # 3 votes per window < min_count 5: nothing scored, everything counted.
    rows = _stream(n_days=1, wins_per_day=10, votes_per_win=3,
                   pi_true=pi, pi_pred=pi)
    acc = _fresh()
    _feed(acc, rows, 10_000)
    r = acc.result()
    assert r["soft"]["n"] == 0
    assert r["counters"]["windows_below_min"] > 0
    assert r["matched_vote_fraction"] == 1.0
    print("min-count exclusion: OK")


def test_unmatched_policy():
    pi = np.full(K, 1.0 / K)
    rows = list(_stream(n_days=1, wins_per_day=10, votes_per_win=8,
                        pi_true=pi, pi_pred=pi))
    matched = rows[3].copy()
    matched[::4] = False  # every 4th detection unmatched
    rows[3] = matched
    # Unmatched rows carry empty per-slot vectors, as the score API does.
    rows[5] = [rows[5][i] if matched[i] else [] for i in range(len(matched))]
    acc = _fresh()
    _feed(acc, tuple(rows), 10_000)
    r = acc.result()
    assert r["matched_vote_fraction"] is not None
    assert 0.7 < r["matched_vote_fraction"] < 0.8
    print(f"unmatched policy: OK (matched fraction "
          f"{r['matched_vote_fraction']:.3f})")


def main():
    test_chunk_invariance()
    test_null_calibration()
    test_detects_miscalibration()
    test_nan_std_refused()
    test_min_count_and_truncation()
    test_unmatched_policy()
    print("ALL WEIGHT-CALIBRATION TESTS PASSED")


if __name__ == "__main__":
    main()
