#!/usr/bin/env python3
r"""Score a baseline's god's-eye preds.csv through KAIROS's online FlowAccumulator.

The rheos/aion per-scene chain fits an accumulated flow model across the training
scenes (each scene loads the previous scene's saved flow state) and, on the final
training scene, scores every held-out TEST detection read-only against that frozen
accumulated model, dumping one row per detection to preds.csv:

    t_ns,ped_id,theta_obs,rho_obs,matched,log_p_joint,log_p_heading,log_p_speed,mean_rho_pred

This re-aggregates those per-detection log-densities with the SAME metric math
KAIROS uses (online_metrics.FlowAccumulator: matched keep their log-density, the
rest get the per-channel floor), so the MLPD is directly comparable to KAIROS's
gods_eye_tbd.txt. No model, no Docker -- pure re-scoring of an existing preds.csv.

    python3 score_baseline_godseye.py --preds results/rheos_perscene/preds.csv \
        --out-dir results/tbd_rheos --label rheos
"""
import argparse
import csv
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from online_metrics import FlowAccumulator, DEFAULT_SPEED_CAP_MPS  # noqa: E402


def _f(s):
    """Parse a CSV cell to float; empty -> NaN."""
    return float(s) if s not in ("", None) else math.nan


def load_preds(path):
    matched, rho_obs = [], []
    lp = {"heading": [], "speed": [], "joint": []}
    mean_rho = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            matched.append(r["matched"] in ("1", "1.0", "True", "true"))
            rho_obs.append(_f(r["rho_obs"]))
            lp["heading"].append(_f(r["log_p_heading"]))
            lp["speed"].append(_f(r["log_p_speed"]))
            lp["joint"].append(_f(r["log_p_joint"]))
            mean_rho.append(_f(r["mean_rho_pred"]))
    return (np.asarray(matched, bool), np.asarray(rho_obs, float),
            {c: np.asarray(v, float) for c, v in lp.items()},
            np.asarray(mean_rho, float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="baseline per-detection preds.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True, help="rheos | aion (for the header)")
    ap.add_argument("--heading-only", action="store_true",
                    help="report heading channel only (aion)")
    ap.add_argument("--speed-cap-mps", type=float, default=DEFAULT_SPEED_CAP_MPS,
                    help="clip observed speeds to this physical pedestrian max (m/s) "
                         "for rho_max/MAE; use the SAME value as evaluate.py "
                         "(0 disables). Precomputed log-densities are not re-clipped.")
    args = ap.parse_args()

    matched, rho_obs, lp, mean_rho = load_preds(args.preds)
    flow = FlowAccumulator(speed_cap_mps=args.speed_cap_mps)
    flow.update(matched, rho_obs, lp, mean_rho)
    f = flow.result()

    def fmt(v):
        return f"{v:.4f}" if v is not None else "--"

    lines = [f"n={f['n_total']} match_rate={f['match_rate']:.4f} rho_max={f['rho_max']:.3f}",
             f"MLPD heading : {fmt(f['mlpd_heading'])}  [matched-only {fmt(f['mlpd_heading_matched_only'])}]"]
    if not args.heading_only:
        lines += [f"MLPD speed   : {fmt(f['mlpd_speed'])}  [matched-only {fmt(f['mlpd_speed_matched_only'])}]",
                  f"MLPD joint   : {fmt(f['mlpd_joint'])}  [matched-only {fmt(f['mlpd_joint_matched_only'])}]",
                  f"Speed MAE    : {fmt(f['speed_mae_mps'])}"]
    if f.get("density_floor"):
        lines += [f"below floor  : H {100*f['below_floor_heading']:.3f}%  "
                  f"S {100*f['below_floor_speed']:.3f}%  J {100*f['below_floor_joint']:.3f}%  "
                  f"of matched (density < {f['density_floor']:g})"]

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines) + "\n"
    (out / "godseye_tbd.txt").write_text(body)
    print(f"[{args.label}] god's-eye (accumulated model, online FlowAccumulator)\n  "
          + "\n  ".join(lines))


if __name__ == "__main__":
    main()
