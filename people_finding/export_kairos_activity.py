r"""Export Kairos's per-(node, slot) presence activity for the
people-finding evaluation. Runs IN THE CONTAINER (pybind pipeline): loads
the frozen embodied state and queries presence_grid at each slot start of
each test day, writing the same A[day, slot, node] table the host sources
produce (nodes = the authored graph's traversable cells; uncovered cells
stay 0, per the own-coverage policy).

    python3 export_kairos_activity.py --graph ... --flow-state ... \
        --scene-dir stub_scene --days 2023-04-24 ... --out /out/A_kairos.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "grounded_sim"))

VOXEL = 0.4
SLOT_S = 600.0
N_SLOTS = int(86400 / SLOT_S)
HORIZON_S = 5.0   # rate-like: p(5s) ~ lambda*5s, unsaturated


def day0_epoch(day_iso):
    from datetime import date
    s = day_iso.replace("-", "")
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    return (date(y, m, d) - date(1970, 1, 1)).days * 86400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--flow-state", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--config-name", default="atc")
    ap.add_argument("--labelspace", default="ade20k_outdoor")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import hydra_python
    from kairos_python.pipeline import load_pipeline
    data = hydra_python.get_dataloader(args.scene_dir)
    pipeline = load_pipeline(data, args.config_name, args.labelspace,
                             output_path=None, enable_lcd=False)
    ok, err = pipeline.load_flow_state(args.flow_state)
    if not ok:
        raise SystemExit(f"load_flow_state failed: {err}")
    from robot_sim import install_nav_layer
    install_nav_layer(pipeline, args.graph)

    doc = json.load(open(args.graph))
    nodes = [n for n in doc["nodes"] if int(n["layer"]) == 3
             and float(n["attributes"].get("distance", 0.0)) > 0]
    pos = np.array([n["attributes"]["position"][:2] for n in nodes])
    kvx = np.floor(pos[:, 0] / VOXEL).astype(np.int64)
    kvy = np.floor(pos[:, 1] / VOXEL).astype(np.int64)
    col_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(kvx, kvy))}
    print(f"[kairos-activity] {len(nodes)} traversable nodes")
    # expose every node cell so presence_grid answers everywhere the state
    # has coverage (the presence protocol's god's-eye exposure step)
    pipeline.set_visible_voxels(
        VOXEL, [[float(pos[i, 0]), float(pos[i, 1]), 0.0]
                for i in range(len(pos))])

    A = np.zeros((len(args.days), N_SLOTS, len(nodes)), dtype=np.float32)
    tic = time.time()
    # presence_grid names each horizon's column p<h>, <h> without a trailing ".0".
    key = f"p{float(HORIZON_S):g}"
    for di, day in enumerate(args.days):
        t0 = day0_epoch(day)
        for si in range(N_SLOTS):
            g = pipeline.presence_grid(t0 + si * SLOT_S, [HORIZON_S])
            gvx = np.asarray(g["vx"], dtype=np.int64)
            gvy = np.asarray(g["vy"], dtype=np.int64)
            p = np.asarray(g[key], dtype=np.float64)
            for i in range(gvx.size):
                c = col_of.get((int(gvx[i]), int(gvy[i])))
                if c is not None:
                    A[di, si, c] = p[i]
        print(f"  [kairos-activity] {day} done ({time.time()-tic:.0f}s)",
              flush=True)
    np.savez_compressed(args.out, A=A, days=np.array(args.days),
                        slot_s=SLOT_S, graph=args.graph)
    print(f"[kairos-activity] wrote {args.out}  A{A.shape} mean {A.mean():.4g}")


if __name__ == "__main__":
    main()
