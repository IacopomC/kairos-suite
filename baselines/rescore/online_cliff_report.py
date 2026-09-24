"""Shared metric aggregation for the online CLiFF-map baseline.

Reports every column of the main flow table (\\Cref{tab:real_results}): coverage,
per-channel MLPD (heading / speed / joint), circular CRPS, MPP and Speed MAE.

The floors are the ones every other method in that table uses, and are pinned
here rather than left to each run:
  * observed speeds clipped at DEFAULT_SPEED_CAP_MPS (3.0 m/s), so a tracker
    glitch cannot move the floor or inflate Speed MAE;
  * the MLPD coverage floor domain pinned to rho_max = 3.000, the value all 66
    existing result files record, so a partial or differently-ordered run stays
    comparable (FlowAccumulator raises if the observed clipped max exceeds it);
  * matched densities below DEFAULT_DENSITY_FLOOR (1e-9) charged at the floor;
  * unmatched detections charged the uniform CRPS pi/4 and MPP 1/8 inside
    GridHeadingScorer.
"""
import math
import pathlib

import numpy as np

from online_metrics import (FlowAccumulator, DEFAULT_SPEED_CAP_MPS,
                            DEFAULT_DENSITY_FLOOR)

# Every current row in the flow table records rho_max=3.000; pinning it keeps
# the coverage floor identical instead of re-deriving it per run.
SHARED_RHO_MAX = 3.0


def new_flow_accumulator():
    return FlowAccumulator(speed_cap_mps=DEFAULT_SPEED_CAP_MPS,
                           rho_max_override=SHARED_RHO_MAX,
                           density_floor=DEFAULT_DENSITY_FLOOR)


class HeadingSums:
    """Running CRPS / MPP totals, all-detections and matched-only."""

    def __init__(self):
        self.crps = self.mpp = self.crps_m = self.mpp_m = 0.0
        self.n = self.n_m = 0

    def add(self, crps, mpp, matched):
        self.crps += float(crps.sum()); self.mpp += float(mpp.sum())
        self.crps_m += float(crps[matched].sum()); self.mpp_m += float(mpp[matched].sum())
        self.n += int(crps.size); self.n_m += int(matched.sum())


def report(label, flow, head, peak_rss_mb, extra=()):
    """Summary lines in the order the flow table reads them."""
    f = flow.result()
    n = max(head.n, 1); nm = max(head.n_m, 1)

    def fmt(v):
        return f"{v:.4f}" if v is not None and np.isfinite(v) else "--"

    lines = [
        f"Fast Online CLiFF-map {label}",
        f"n={head.n} n_matched={head.n_m} match_rate={head.n_m/n:.4f} "
        f"rho_max={f['rho_max']:.3f} (pinned {SHARED_RHO_MAX:.3f}, cap {DEFAULT_SPEED_CAP_MPS:.1f} m/s)",
        f"MLPD heading : {fmt(f['mlpd_heading'])}  [matched-only {fmt(f['mlpd_heading_matched_only'])}]",
        f"MLPD speed   : {fmt(f['mlpd_speed'])}  [matched-only {fmt(f['mlpd_speed_matched_only'])}]",
        f"MLPD joint   : {fmt(f['mlpd_joint'])}  [matched-only {fmt(f['mlpd_joint_matched_only'])}]",
        f"circular CRPS: {head.crps/n:.4f}  [matched-only {head.crps_m/nm:.4f}]  (uniform {math.pi/4:.4f})",
        f"MPP          : {head.mpp/n:.4f}  [matched-only {head.mpp_m/nm:.4f}]  (uniform 0.1250)",
        f"Speed MAE    : {fmt(f['speed_mae_mps'])} m/s",
    ]
    if f.get("density_floor"):
        lines.append(
            f"below floor  : H {100*f['below_floor_heading']:.3f}%  "
            f"S {100*f['below_floor_speed']:.3f}%  J {100*f['below_floor_joint']:.3f}%  "
            f"of matched (density < {f['density_floor']:g})")
    lines += list(extra)
    lines.append(f"peak RSS: {peak_rss_mb:.0f} MB")
    return lines


def write(out_path, lines):
    body = "\n".join(lines) + "\n"
    print(body, flush=True)
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
