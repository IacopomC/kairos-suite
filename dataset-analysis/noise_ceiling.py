"""TBD heading-noise ceiling: how much of the heading score is measurement
noise rather than forecast error?

The held-out tracks' headings come from 10 Hz tracking; frame-to-frame
jitter makes them noisy. This pass smooths each person's positions over
time, recomputes their headings from the smoothed motion, and scores the
SAME saved predictions against raw and smoothed headings. The improvement
against smoothed headings is the part of the error the data itself
carries: an estimate of the noise ceiling the paper's noise-limited
reading refers to.

Gate: recomputing the heading log-density from the saved slot weights must
reproduce the stored per-row log_p_heading before smoothed headings are
scored.

    python3 noise_ceiling.py --preds path/to/preds.csv

Usage:
    python3 noise_ceiling.py --preds results/<run>/preds.csv

Scores the same saved predictions twice, against raw and against
time-smoothed headings. The gap is measurement noise rather than forecast
error, so it bounds how much of the remaining heading score any model could
recover on this dataset.

The smoothing constants are module-level: GAP_S splits a track across a
sampling gap, SMOOTH_S is the position half-window, DIFF_S the
central-difference half-window. They assume roughly 10 Hz tracking; raise them
for a slower stream.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SIGMA = 0.4
SLOTS = np.arange(8) * (np.pi / 4)
GAP_S = 0.5          # split a track where consecutive samples are further apart
SMOOTH_S = 0.25      # centered moving-average half-window on positions
DIFF_S = 0.2         # central-difference half-window for the heading


def wrapped_logpdf(theta, weights):
    """log of the semi-wrapped mixture density at theta (rows aligned)."""
    d = theta[:, None] - SLOTS[None, :]
    p = np.zeros_like(d)
    for w in (-2, -1, 0, 1, 2):
        p += np.exp(-0.5 * ((d + 2 * np.pi * w) / SIGMA) ** 2)
    p /= (SIGMA * np.sqrt(2 * np.pi))
    return np.log(np.maximum((weights * p).sum(axis=1), 1e-300))


def sector_mass_matrix():
    G = 14400
    ang = (np.arange(G) + 0.5) * (2 * np.pi / G)
    A = np.zeros((8, 8))
    sec = (np.round(ang / (np.pi / 4)).astype(int)) % 8
    for k in range(8):
        d = np.angle(np.exp(1j * (ang - SLOTS[k])))
        pdf = np.zeros(G)
        for w in (-2, -1, 0, 1, 2):
            pdf += np.exp(-0.5 * ((d + 2 * np.pi * w) / SIGMA) ** 2)
        pdf /= pdf.sum()
        for s in range(8):
            A[k, s] = pdf[sec == s].sum()
    return A


def smooth_headings(df):
    """Per-person smoothed headings; NaN where a segment is too short."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby("ped_id"):
        g = g.sort_values("t_ns")
        idx = g.index.to_numpy()
        t = g["t_ns"].to_numpy() / 1e9
        x = g["x_world"].to_numpy()
        y = g["y_world"].to_numpy()
        seg = np.concatenate([[0], np.cumsum(np.diff(t) > GAP_S)])
        for sid in np.unique(seg):
            m = seg == sid
            if m.sum() < 5:
                continue
            ts, xs, ys, ii = t[m], x[m], y[m], idx[m]
            xs_s = np.array([xs[(ts >= ti - SMOOTH_S) & (ts <= ti + SMOOTH_S)].mean()
                             for ti in ts])
            ys_s = np.array([ys[(ts >= ti - SMOOTH_S) & (ts <= ti + SMOOTH_S)].mean()
                             for ti in ts])
            for j, ti in enumerate(ts):
                lo = np.searchsorted(ts, ti - DIFF_S, "left")
                hi = np.searchsorted(ts, ti + DIFF_S, "right") - 1
                if hi > lo:
                    dx = xs_s[hi] - xs_s[lo]
                    dy = ys_s[hi] - ys_s[lo]
                    if dx * dx + dy * dy > 1e-8:
                        out[np.where(idx == ii[j])[0][0] if False else
                            df.index.get_loc(ii[j])] = np.arctan2(dy, dx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    wcols = [f"slot_weight_{k}" for k in range(8)]
    m = (df["matched"] == 1) & (df[wcols].sum(axis=1) > 0.5)
    d = df[m].reset_index(drop=True)
    W = d[wcols].to_numpy()
    W = W / W.sum(axis=1, keepdims=True)
    theta_raw = np.mod(d["theta_obs"].to_numpy(), 2 * np.pi)

    # gate: reproduce the stored heading log-density
    lp = wrapped_logpdf(theta_raw, W)
    err = np.abs(lp - d["log_p_heading"].to_numpy())
    print(f"[gate] recomputed vs stored log_p_heading on {len(d):,} matched "
          f"rows: median |err| {np.median(err):.2e}, p99 "
          f"{np.percentile(err, 99):.2e}")
    assert np.median(err) < 1e-3, "density formula does not match the scorer"

    theta_s = smooth_headings(d)
    ok = ~np.isnan(theta_s)
    print(f"[smooth] headings recomputed for {ok.sum():,} of {len(d):,} rows "
          f"({ok.mean():.3f})")

    A = sector_mass_matrix()
    sec_raw = (np.round(theta_raw / (np.pi / 4)).astype(int)) % 8
    sec_s = (np.round(np.mod(theta_s[ok], 2 * np.pi) / (np.pi / 4)).astype(int)) % 8
    masses = W @ A

    lp_raw = wrapped_logpdf(theta_raw[ok], W[ok])
    lp_s = wrapped_logpdf(np.mod(theta_s[ok], 2 * np.pi), W[ok])
    mpp_raw = masses[ok, :][np.arange(ok.sum()), sec_raw[ok]]
    mpp_s = masses[ok, :][np.arange(ok.sum()), sec_s]

    print("\nsame predictions, raw vs smoothed observed headings "
          f"(n={ok.sum():,}):")
    print(f"  heading MLPD : raw {lp_raw.mean():+.4f}   smoothed "
          f"{lp_s.mean():+.4f}   ceiling {lp_s.mean()-lp_raw.mean():+.4f} nats")
    print(f"  MPP          : raw {mpp_raw.mean():.4f}   smoothed "
          f"{mpp_s.mean():.4f}")
    print(f"  raw-vs-smoothed heading difference: median "
          f"{np.degrees(np.median(np.abs(np.angle(np.exp(1j*(theta_raw[ok]-theta_s[ok])))))):.1f} deg")


if __name__ == "__main__":
    main()
