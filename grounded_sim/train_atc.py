#!/usr/bin/env python3
r"""TRAIN the Kairos flow model on the ATC simulation.

ATC has no sensor stream, so a simulated robot substitutes for the reconstruction
frontend: it patrols the given map and, per 10 Hz keyframe, reports its 5 m
active-window sphere and the detections inside it, fed to the pipeline exactly as
the live system would:
    set_visible_voxels(0.4, <walkable voxels within 5 m>)   # == handleMapUpdate
    push_detections(t, <detections within 5 m>, theta, rho)
    flush_detections()                                      # process THIS frame
Training runs over one or more TRAIN days in sequence through a single pipeline,
so the flow state accumulates on absolute unix time (this is what lets the NUDFT
recover daily/weekly rhythm across sessions). The result is a saved flow_state.bin.
No reconstruction frontend runs; the stub scene only supplies the sensor object
`from_config` requires. This is the ATC (simulation) counterpart of `kairos_python
run --tracks` for real-sensor datasets.

Odometry is exact by default. With a --drift-config carrying a nonzero rate the
platform instead builds its map in a drifted frame: its position error displaces
every visible cell and detection it registers, and its heading error turns them
about its own position and turns each detection's bearing with them. Each
completed patrol loop fires a simulated loop closure, and --deform selects
whether the correction warp re-keys the accumulated state (on) or the state
stays on its drifted keys (off). That regime additionally writes per-day
prequential scores and lc_events.json. The rates live in
kairos-suite/config/drift/, one file per run.

Usage (in the kairos Docker image):
    python3 train_atc.py --graph <atc_graph.json> --days <atc-*.npz ...> \
        --scene-dir stub_scene --config-name atc \
        --out-dir OUT/train --out-state OUT/train/flow_state.bin
Then score the held-out days with evaluation/evaluate.py, or run both via train_test.sh.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import numpy as np

from scene_loader import get_dataloader
from kairos_python.pipeline import load_pipeline

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from robot_sim import (load_nav_graph, load_nav_clearance, build_coverage_loop,
                       RobotSim, KEYFRAME_DT_S, ROBOT_SPEED_MPS,
                       ACTIVE_WINDOW_RADIUS_M)
from atc_stream import AtcStream

# Keyframes between place-summary aggregation passes. The pass lifts voxel
# flow onto the navigational layer and applies the traversable-adjacency
# neighbourhood to the coupling, so it has to run during training, not only
# at readout.
AGGREGATE_EVERY_KF = 5000

VOXEL_SIZE = 0.4


class DriftModel:
    """Seeded odometric drift: the magnitude grows deterministically with
    distance traveled (`per_loop_m / loop_length_m` per metre, so exactly
    per_loop_m at each correction), along a direction that wanders as a
    random walk; both reset at each correction.

    The heading estimate drifts on the same footing: `yaw_per_loop_rad` of
    error accumulated per completed loop, growing linearly with distance
    traveled and resetting with the translation. A yaw error rotates every
    bearing the platform registers about its own position, so its effect on
    the map grows with the distance from the platform, unlike the translation.
    """

    def __init__(self, per_loop_m, loop_length_m, seed, angle_sigma=0.02,
                 yaw_per_loop_rad=0.0):
        self.rate = per_loop_m / loop_length_m if per_loop_m > 0 else 0.0
        self.yaw_rate = (yaw_per_loop_rad / loop_length_m
                         if yaw_per_loop_rad else 0.0)
        self.rng = np.random.default_rng(seed)
        self.angle = self.rng.uniform(0.0, 2.0 * np.pi)
        self.angle_sigma = angle_sigma
        self.s = 0.0
        self.d = np.zeros(2)
        self.psi = 0.0

    def step(self, ds):
        if self.rate == 0.0 and self.yaw_rate == 0.0:
            return
        self.s += ds
        self.angle += self.rng.normal(0.0, self.angle_sigma)
        self.d = self.rate * self.s * np.array([np.cos(self.angle), np.sin(self.angle)])
        self.psi = self.yaw_rate * self.s

    def reset(self):
        self.s = 0.0
        self.d = np.zeros(2)
        self.psi = 0.0
        self.angle = self.rng.uniform(0.0, 2.0 * np.pi)

    def register(self, xy, pose_xy):
        """Where the platform records world points @p xy (N x 2), given its
        drifted pose estimate. The heading error rotates the measurement about
        the platform's own position; the translation error displaces it."""
        if self.yaw_rate == 0.0:
            return xy + self.d
        c, s = np.cos(self.psi), np.sin(self.psi)
        rel = xy - pose_xy
        rotated = np.column_stack((c * rel[:, 0] - s * rel[:, 1],
                                   s * rel[:, 0] + c * rel[:, 1]))
        return pose_xy + rotated + self.d


# Built-in drift settings, used for every entry a drift config leaves out and
# when no config is given at all. Zero rates are the drift-free regime.
DRIFT_DEFAULTS = {
    "per_loop_m": 0.0,
    "yaw_per_loop_deg": 0.0,
    "seed": 7,
    "control_spacing_s": 5.0,
}


def load_drift_config(path, overrides):
    """Resolve the drift model from its config file and the CLI overrides.
    Every rate lives in the file; an override is for a one-off sweep."""
    settings = dict(DRIFT_DEFAULTS)
    if path:
        import yaml
        with open(path) as handle:
            loaded = yaml.safe_load(handle) or {}
        unknown = set(loaded) - set(DRIFT_DEFAULTS)
        if unknown:
            raise SystemExit(f"[run_atc] drift config {path} has unknown "
                             f"entries: {sorted(unknown)}")
        settings.update(loaded)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings


class DriftRun:
    """State carried across days by the drifted-odometry regime: the drift
    model, the warp control-point history, and the loop-closure event log.
    The patrol phase is continuous across days here, unlike the drift-free
    regime which restarts each day from the day's own t0."""

    def __init__(self, drift, deform, drift_reset, ctrl_every, out_dir,
                 viz_dir=None, viz_field_every=10):
        self.drift = drift
        self.out_dir = pathlib.Path(out_dir)
        self.deform = deform
        self.drift_reset = drift_reset
        self.ctrl_every = ctrl_every
        self.viz_dir = viz_dir
        self.viz_field_every = viz_field_every
        self.hist_t, self.hist_true, self.hist_drift = [], [], []
        self.hist_yaw = []
        self.events = []
        self.arclength = 0.0
        self.loop_idx = 0
        self.rows_t, self.rows_m, self.rows_j, self.rows_h = [], [], [], []

    def start_day(self):
        if self.drift_reset == "day":
            self.drift.reset()
        self.rows_t, self.rows_m, self.rows_j, self.rows_h = [], [], [], []


def dump_viz(pipeline, tag, t, out_dir, node_ids, include_field=True):
    """Node summaries at time `t` (and optionally the voxel field) -> one npz."""
    pipeline.update_place_summaries(t)
    s = pipeline.place_summaries()
    payload = dict(
        t=t, node_ids=np.asarray(node_ids),
        node_prob=np.asarray(s["probability"]),
        node_dom_theta=np.asarray(s["dominant_theta"]),
        node_dom_rho=np.asarray(s["dominant_rho"]),
        node_support=np.asarray(s["support"]))
    if include_field:
        vox = pipeline.flow_voxels()
        payload["field_n"] = np.asarray([len(vox)])
    np.savez_compressed(pathlib.Path(out_dir) / f"{tag}.npz", **payload)


def drive_day(pipeline, sim, pos, day_path, x_bounds, y_bounds, minutes,
              gods_eye=False, decimate=1, stream=None,
              active_mask=None, decide_dt_s=None, speed=ROBOT_SPEED_MPS,
              drift_run=None, node_ids=None):
    """Replay one ATC day, feeding the pipeline per 10 Hz keyframe. Embodied
    mode (default): the simulated robot patrols and only its 5 m active window
    is visible per keyframe. God's-eye mode: every mapped cell is visible for
    the whole day (set once by the caller) and every detection is fed --- the
    full-observability regime the grid baselines are fitted in. `decimate` (>1)
    keeps every Nth detection, uniformly; the estimators are running averages
    and spectra, so decimation thins counts without biasing means. Returns
    detections pushed. The patrol restarts from the day's t0; cross-day
    periodicity is recovered by the NUDFT on absolute unix time regardless of
    the per-day patrol phase."""
    if drift_run is not None:
        drift_run.start_day()
        ds = speed * KEYFRAME_DT_S
    if stream is None:
        stream = AtcStream(day_path, x_bounds=x_bounds, y_bounds=y_bounds)
    t0 = stream.t_min
    dur = (stream.t_max - t0) if minutes == 0 else minutes * 60.0
    n_kf = int(dur / KEYFRAME_DT_S)
    r2 = sim.radius ** 2
    print(f"[run_atc]   {pathlib.Path(day_path).stem}: {n_kf} keyframes "
          f"({dur/60:.1f} min) ...")
    pushed = 0
    tic = time.time()
    for k in range(n_kf):
        t = t0 + k * KEYFRAME_DT_S
        # exploration ratio: outside the active intervals the robot is occupied
        # elsewhere and observes nothing (empty active window, no detections).
        observing = True
        if not gods_eye and active_mask is not None:
            iv = int((k * KEYFRAME_DT_S) / decide_dt_s)
            observing = iv < len(active_mask) and bool(active_mask[iv])
        if not gods_eye:
            if observing:
                if drift_run is None:
                    # Drift-free: the patrol phase restarts from the day's t0.
                    pose = sim.pose_at_arclength(speed * k * KEYFRAME_DT_S)
                    offset = None
                else:
                    # Drifted odometry: the phase is continuous across days, and
                    # everything the platform registers is displaced by the
                    # current drift, so the map is built in the drifted frame.
                    pose = sim.pose_at_arclength(
                        drift_run.arclength % sim.total)
                    drift_run.drift.step(ds)
                    offset = drift_run.drift.d
                    if k % drift_run.ctrl_every == 0:
                        drift_run.hist_t.append(t)
                        drift_run.hist_true.append(
                            [float(pose[0]), float(pose[1]), 0.0])
                        drift_run.hist_drift.append(
                            [float(offset[0]), float(offset[1])])
                        drift_run.hist_yaw.append(float(drift_run.drift.psi))
                vis = sim.visible_places(pose)
                if offset is None:
                    vis_xyz = [[float(pos[i, 0]), float(pos[i, 1]), 0.0]
                               for i in vis]
                else:
                    seen = drift_run.drift.register(
                        np.asarray([[pos[i, 0], pos[i, 1]] for i in vis],
                                   dtype=float).reshape(-1, 2),
                        np.asarray(pose[:2], dtype=float))
                    vis_xyz = [[float(x), float(y), 0.0] for x, y in seen]
                pipeline.set_visible_voxels(VOXEL_SIZE, vis_xyz)
            else:
                pipeline.set_visible_voxels(VOXEL_SIZE, [])

        if not observing:
            pipeline.flush_detections()
            if k % AGGREGATE_EVERY_KF == 0 and k:
                pipeline.update_place_summaries(t)
            if k % 20000 == 0 and k:
                print(f"    kf {k}/{n_kf}  pushed={pushed}  "
                      f"({k/(time.time()-tic):.0f} kf/s)")
            continue

        dx, dy, th, rh, pid = stream.between(t, t + KEYFRAME_DT_S)
        if len(dx):
            if gods_eye:
                if decimate > 1:
                    m = (np.arange(len(dx)) + k) % decimate == 0
                    dx, dy, th, rh, pid = dx[m], dy[m], th[m], rh[m], pid[m]
            else:
                m = ((dx - pose[0]) ** 2 + (dy - pose[1]) ** 2) <= r2
                dx, dy, th, rh, pid = dx[m], dy[m], th[m], rh[m], pid[m]
        if drift_run is None or gods_eye:
            det_xyz = [[float(x), float(y), 0.0] for x, y in zip(dx, dy)]
        else:
            seen = drift_run.drift.register(
                np.column_stack((dx, dy)).astype(float).reshape(-1, 2),
                np.asarray(pose[:2], dtype=float))
            det_xyz = [[float(x), float(y), 0.0] for x, y in seen]
            # A heading is a bearing: it turns with the platform's heading error.
            th = np.mod(np.asarray(th, dtype=float) + drift_run.drift.psi,
                        2.0 * np.pi)
        if drift_run is not None and det_xyz:
            # Prequential: score the perceived detections before learning them.
            sc = pipeline.score_observations(
                timestamps_s=[float(t)] * len(det_xyz), xyz_world=det_xyz,
                theta=[float(v) for v in th], rho=[float(v) for v in rh],
                horizon_s=0.0)
            drift_run.rows_t.append(np.full(len(det_xyz), t))
            drift_run.rows_m.append(np.asarray(sc["matched"], dtype=bool))
            drift_run.rows_j.append(
                np.asarray(sc["log_p_joint"], dtype=np.float32))
            drift_run.rows_h.append(np.asarray(
                sc.get("log_p_heading", [np.nan] * len(det_xyz)),
                dtype=np.float32))
        pipeline.push_detections(t, det_xyz,
                                 [float(v) for v in th], [float(v) for v in rh],
                                 track_ids=[int(v) for v in pid])
        pipeline.flush_detections()
        pushed += len(det_xyz)
        if drift_run is not None:
            drift_run.arclength += ds
            if int(drift_run.arclength / sim.total) > drift_run.loop_idx:
                # Loop completed: a simulated loop closure. The pose estimate
                # snaps to the true trajectory; with --deform on it additionally
                # re-keys the accumulated state through the correction warp.
                drift_run.loop_idx = int(drift_run.arclength / sim.total)
                drift_before = float(np.hypot(*drift_run.drift.d))
                n_ev = len(drift_run.events)
                if drift_run.viz_dir is not None:
                    dump_viz(pipeline, f"e{n_ev:04d}_pre", t, drift_run.viz_dir,
                             node_ids,
                             include_field=n_ev % drift_run.viz_field_every == 0)
                if drift_run.deform == "on" and drift_run.drift.rate > 0.0:
                    before = [[tp[0] + dd[0], tp[1] + dd[1], tp[2]]
                              for tp, dd in zip(drift_run.hist_true,
                                                drift_run.hist_drift)]
                    # The map holds content registered under the drifted
                    # estimate, so the correction turns it back by -psi.
                    yaw_after = [-float(v) for v in drift_run.hist_yaw]
                    pipeline.apply_deformation_snapshot(
                        float(t), before,
                        [float(v) for v in drift_run.hist_t],
                        [list(map(float, tp)) for tp in drift_run.hist_true],
                        yaw_after)
                if drift_run.drift_reset == "loop":
                    drift_run.drift.reset()
                drift_run.events.append(
                    {"t": float(t), "loop": drift_run.loop_idx,
                     "drift_before_m": drift_before,
                     "n_control_points": len(drift_run.hist_t)})
                if drift_run.viz_dir is not None:
                    dump_viz(pipeline, f"e{n_ev:04d}_post", t,
                             drift_run.viz_dir, node_ids,
                             include_field=n_ev % drift_run.viz_field_every == 0)
                if len(drift_run.events) % 10 == 0:
                    print(f"    event {len(drift_run.events)} "
                          f"(loop {drift_run.loop_idx}) "
                          f"drift_before={drift_before:.2f} m", flush=True)
        if k % AGGREGATE_EVERY_KF == 0 and k:
            pipeline.update_place_summaries(t)
        if k % 20000 == 0 and k:
            print(f"    kf {k}/{n_kf}  pushed={pushed}  "
                  f"({k/(time.time()-tic):.0f} kf/s)")
    if drift_run is not None:
        out = drift_run.out_dir
        r = drift_run
        np.savez_compressed(
            out / f"rows_{pathlib.Path(day_path).stem}.npz",
            t=np.concatenate(r.rows_t) if r.rows_t else np.array([]),
            matched=np.concatenate(r.rows_m) if r.rows_m
            else np.array([], bool),
            log_p_joint=np.concatenate(r.rows_j) if r.rows_j
            else np.array([], np.float32),
            log_p_heading=np.concatenate(r.rows_h) if r.rows_h
            else np.array([], np.float32))
        print(f"[run_atc]   done day: {pushed} detections in "
              f"{time.time()-tic:.0f}s ({len(r.events)} events so far)",
              flush=True)
        return pushed
    print(f"[run_atc]   done day: {pushed} detections in {time.time()-tic:.0f}s")
    return pushed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--days", nargs="+", required=True,
                    help="one or more atc-YYYYMMDD.npz/.csv training days, driven "
                         "in sequence through one pipeline (state accumulates)")
    ap.add_argument("--scene-dir", required=True, help="stub scene (sensor only)")
    ap.add_argument("--config-name", default="atc")
    ap.add_argument("--labelspace", default="ade20k_outdoor")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-state", required=True)
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="0 = the whole session per day")
    ap.add_argument("--fov-m", type=float, default=ACTIVE_WINDOW_RADIUS_M,
                    help="active-window (field-of-view) radius for the embodied "
                         "robot. The config must carry a matching "
                         "map_window.max_radius_m or the C++ archives cells the "
                         "sensor can still see. Ignored with --gods-eye")
    ap.add_argument("--gods-eye", action="store_true",
                    help="full-observability training: every mapped cell visible "
                         "all day, every detection fed (the grid baselines' "
                         "regime); no robot patrol")
    ap.add_argument("--decimate", type=int, default=1,
                    help="god's-eye only: keep every Nth detection (uniform "
                         "thinning; running-average/spectral estimators are "
                         "unbiased in expectation)")
    ap.add_argument("--patrol-policy", default="serpentine",
                    choices=["serpentine", "random"],
                    help="patrol: 'serpentine' is the fixed coverage loop (the "
                         "protocol default); 'random' picks the observed region "
                         "uniformly per 10-min interval (the sparse-coverage "
                         "study).")
    ap.add_argument("--policy-seed", type=int, default=7)
    ap.add_argument("--policy-regions", type=int, default=24)
    ap.add_argument("--explore-ratio", type=float, default=1.0,
                    help="Sparse-coverage observation budget: the "
                         "fraction of the day's 10-min decision intervals the "
                         "robot spends observing (the rest it is occupied "
                         "elsewhere). Active intervals are spread across the "
                         "day by even stride. 1.0 = observe all day (default); "
                         "applies to the region patrol, not serpentine/god's-eye")
    ap.add_argument("--robot-speed", type=float, default=ROBOT_SPEED_MPS,
                    help="robot travel speed (m/s); default matches the "
                         "main-results serpentine runs")
    ap.add_argument("--stream-source", default="atc", choices=["atc", "hb"],
                    help="'hb': --days are ISO dates of a station stream, "
                         "interpolated per track to --interp-hz")
    ap.add_argument("--hb-root", default="/data/hb-station/Data-2")
    ap.add_argument("--station", choices=["HB", "PB"], default="HB")
    ap.add_argument("--interp-hz", type=float, default=10.0)
    ap.add_argument("--gap-max-s", type=float, default=2.0)
    ap.add_argument("--drift-config", default=None, metavar="YAML",
                    help="drift model to inject: a YAML holding per_loop_m and "
                         "yaw_per_loop_deg (see kairos-suite/config/drift/). "
                         "Both rates come from the file so a run is changed by "
                         "editing it, not by editing a launch script. The "
                         "--drift-per-loop-m / --drift-yaw-per-loop-deg flags "
                         "override individual entries")
    ap.add_argument("--drift-per-loop-m", type=float, default=None,
                    help="translational odometric drift accumulated per "
                         "completed patrol loop (metres), overriding the drift "
                         "config. 0 with no yaw is the drift-free regime: no "
                         "drift, no loop-closure events, no correction warps. "
                         "Above 0 the platform builds its map in the drifted "
                         "frame and each completed loop fires a simulated loop "
                         "closure")
    ap.add_argument("--drift-yaw-per-loop-deg", type=float, default=None,
                    help="heading (yaw) drift accumulated per completed patrol "
                         "loop (degrees), overriding the drift config. A yaw "
                         "error turns every bearing the platform registers "
                         "about its own position, so it displaces the map in "
                         "proportion to the distance from the trajectory")
    ap.add_argument("--deform", choices=["on", "off"], default="on",
                    help="with drift enabled, deliver the correction warp at "
                         "each loop-closure event (on) or leave the state on "
                         "its drifted keys (off)")
    ap.add_argument("--drift-reset", choices=["loop", "day"], default="loop",
                    help="when the pose estimate re-anchors to the truth: at "
                         "every loop event (a robot whose loop closures also "
                         "fix its localization) or only at the start of each "
                         "day (no loop closure: the error compounds all day "
                         "and the robot starts each morning re-localized)")
    ap.add_argument("--drift-seed", type=int, default=None)
    ap.add_argument("--control-spacing-s", type=float, default=5.0,
                    help="spacing of warp control points along the trajectory")
    ap.add_argument("--viz", action="store_true",
                    help="with drift enabled, dump node summaries pre/post each "
                         "loop-closure event, plus the voxel field every "
                         "--viz-field-every events")
    ap.add_argument("--viz-field-every", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=0, metavar="N",
                    help="write a crash-recovery checkpoint after every N days "
                         "(0 = off). The state and a manifest recording how many "
                         "days it covers go to <out-dir>/ckpt/. With --resume a "
                         "later invocation restarts from the newest one instead "
                         "of replaying the whole corpus.")
    ap.add_argument("--resume", action="store_true",
                    help="load the newest checkpoint in <out-dir>/ckpt/ and skip "
                         "the days it already covers. Requires the same --days "
                         "list: the manifest records it and a mismatch aborts.")
    ap.add_argument("--checkpoint-before", nargs="*", default=[], metavar="DATE:PATH",
                    help="save the accumulated flow state just before the first day "
                         "on/after DATE (ISO), e.g. 2023-01-01:/out/state_jan.bin. "
                         "Repeatable; yields as-of-date states from a single pass so "
                         "a window can be visualised with its inherited model.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    data = get_dataloader(args.scene_dir)
    pipeline = load_pipeline(data, args.config_name, args.labelspace,
                             output_path=out_dir, enable_lcd=False)
    if pipeline is None:
        print("[run_atc] failed to load pipeline"); return

    pos, node_ids, adj, trav = load_nav_graph(args.graph)
    poly, order = build_coverage_loop(pos, adj, trav)
    sim = RobotSim(poly, pos, radius=args.fov_m)
    print(f"[run_atc] {len(pos)} cells ({int(trav.sum())} traversable), "
          f"serpentine coverage loop {sim.total:.0f} m "
          f"({sim.total/ROBOT_SPEED_MPS/60:.1f} min/loop), {len(args.days)} train days")
    xb = (pos[:, 0].min() - 6, pos[:, 0].max() + 6)
    yb = (pos[:, 1].min() - 6, pos[:, 1].max() + 6)

    # Install the authored graph as Kairos's navigational layer: node centres,
    # per-node authored clearance, and the traversable edges. The coupling's
    # neighbourhood is read from this layer, so the edges and the clearances
    # must both be present. set_visible_voxels installs the voxel grid and has
    # to precede the layer.
    clearance = load_nav_clearance(args.graph)
    nav_edges = sorted({(min(i, j), max(i, j))
                        for i, j in zip(*adj.nonzero()) if i != j})
    pipeline.set_visible_voxels(VOXEL_SIZE, [])
    nav_node_ids = pipeline.set_navigational_layer(
        [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(len(pos))],
        [float(c) for c in clearance],
        [[int(i), int(j)] for i, j in nav_edges])
    print(f"[run_atc] nav layer: {len(pos)} nodes, {len(nav_edges)} edges, "
          f"clearance median {float(np.median(clearance)):.2f} m, "
          f"{int((clearance <= 0).sum())} nodes at zero clearance")

    drift = load_drift_config(args.drift_config, {
        "per_loop_m": args.drift_per_loop_m,
        "yaw_per_loop_deg": args.drift_yaw_per_loop_deg,
        "seed": args.drift_seed,
        "control_spacing_s": args.control_spacing_s,
    })

    drift_run = None
    if drift["per_loop_m"] > 0.0 or drift["yaw_per_loop_deg"] != 0.0:
        if args.gods_eye:
            raise SystemExit("[run_atc] --gods-eye and drift are exclusive: "
                             "full visibility has no patrol to drift")
        if args.patrol_policy != "serpentine":
            raise SystemExit("[run_atc] drift requires --patrol-policy "
                             "serpentine: the loop-closure event is a completed "
                             "coverage loop")
        viz_dir = None
        if args.viz:
            viz_dir = out_dir / "viz"; viz_dir.mkdir(exist_ok=True)
        drift_run = DriftRun(
            DriftModel(drift["per_loop_m"], sim.total, drift["seed"],
                       yaw_per_loop_rad=np.radians(drift["yaw_per_loop_deg"])),
            args.deform, args.drift_reset,
            max(1, int(drift["control_spacing_s"] / KEYFRAME_DT_S)),
            out_dir=out_dir,
            viz_dir=viz_dir, viz_field_every=args.viz_field_every)
        print(f"[run_atc] drifted odometry: {drift['per_loop_m']} m/loop, "
              f"{drift['yaw_per_loop_deg']} deg/loop yaw, "
              f"seed {drift['seed']}, deform={args.deform}, "
              f"reset={args.drift_reset}, control spacing "
              f"{drift['control_spacing_s']:g}s"
              + (f" (from {args.drift_config})" if args.drift_config else ""))

    if args.gods_eye:
        # Full observability, once: every mapped cell stays visible (presence
        # exposure accrues map-wide per frame) and detections are never gated
        # by a pose.
        pipeline.set_visible_voxels(
            VOXEL_SIZE, [[float(pos[i, 0]), float(pos[i, 1]), 0.0]
                         for i in range(len(pos))])
        print(f"[run_atc] GOD'S-EYE regime: all {len(pos)} cells visible, "
              f"decimate={args.decimate}")

    stream_open = None
    if args.stream_source == "hb":
        from hb_stream import HBDayStream, hb_day_files

        def stream_open(day):
            files = hb_day_files(args.hb_root, args.station, day)
            if not files:
                raise SystemExit(f"[run_atc] no hb files for {day}")
            return HBDayStream(files, x_bounds=xb, y_bounds=yb,
                                  interp_hz=args.interp_hz,
                                  gap_max_s=args.gap_max_s)

    policy = args.patrol_policy
    if policy != "serpentine" and not args.gods_eye:
        from explore_patrol import (DECIDE_DT_S, RegionPatrols, active_intervals,
                                    day_polyline, make_regions, sample_schedule)
        labels, _ = make_regions(pos, trav, k=args.policy_regions,
                                 seed=args.policy_seed)
        patrols = RegionPatrols(pos, adj, trav, labels, args.policy_regions)
        rng = np.random.default_rng(args.policy_seed)
        print(f"[run_atc] policy '{policy}': {args.policy_regions} regions, "
              f"decisions every {DECIDE_DT_S:.0f}s")

    total = 0
    pending_ckpt = [tuple(c.split(":", 1)) for c in args.checkpoint_before]

    # Crash recovery. The manifest pins the day list the checkpoint was built
    # from, so resuming against a different corpus fails loudly instead of
    # silently producing a state trained on the wrong days.
    ckpt_dir = out_dir / "ckpt"
    ckpt_state = ckpt_dir / "flow_state.bin"
    ckpt_manifest = ckpt_dir / "manifest.json"
    days_key = [str(d) for d in args.days]
    start_at = 0
    if args.resume and ckpt_manifest.exists() and ckpt_state.exists():
        with open(ckpt_manifest) as f:
            man = json.load(f)
        if man.get("days") != days_key:
            raise SystemExit(f"[run_atc] --resume: {ckpt_manifest} was built from a "
                             f"different --days list ({len(man.get('days', []))} days "
                             f"vs {len(days_key)} requested); refusing to resume")
        ok, err = pipeline.load_flow_state(str(ckpt_state))
        if not ok:
            raise SystemExit(f"[run_atc] --resume: load_flow_state({ckpt_state}): {err!r}")
        start_at = int(man["days_done"])
        total = int(man.get("detections", 0))
        print(f"[run_atc] resumed from {ckpt_state}: {start_at}/{len(days_key)} days "
              f"already accumulated ({total} detections); continuing at "
              f"{days_key[start_at] if start_at < len(days_key) else 'end'}", flush=True)
    elif args.resume:
        print(f"[run_atc] --resume: no checkpoint at {ckpt_state}, starting from "
              f"the first day", flush=True)

    def write_checkpoint(days_done, detections):
        """Save state plus the manifest --resume reads, atomically enough that a
        kill mid-write cannot leave a manifest pointing at a truncated state."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_dir / "flow_state.bin.part"
        ok, err = pipeline.save_flow_state(str(tmp))
        if not ok:
            print(f"[run_atc] checkpoint FAILED at day {days_done}: {err!r}", flush=True)
            return
        os.replace(tmp, ckpt_state)  # state lands before the manifest names it
        with open(ckpt_manifest, "w") as f:
            json.dump({"days_done": days_done, "detections": detections,
                       "days": days_key}, f)
        print(f"[run_atc] checkpoint: {days_done}/{len(days_key)} days -> "
              f"{ckpt_state}", flush=True)

    for di, day in enumerate(args.days):
        if di < start_at:
            continue
        for date, path in list(pending_ckpt):
            if str(day) >= date:
                ok, err = pipeline.save_flow_state(path)
                print(f"[run_atc] checkpoint as-of {date} (before day {day}, "
                      f"{di} days accumulated): ok={ok} err={err!r} -> {path}", flush=True)
                pending_ckpt.remove((date, path))
        stream = stream_open(day) if stream_open else None
        if policy != "serpentine" and not args.gods_eye:
            if stream is None:
                stream = AtcStream(day, x_bounds=xb, y_bounds=yb)
            dur = stream.t_max - stream.t_min
            n_iv = max(1, int(np.ceil(dur / DECIDE_DT_S)))
            w = np.ones(args.policy_regions)
            schedule = sample_schedule(w, n_iv, rng)
            sim = RobotSim(day_polyline(patrols, schedule, dur,
                                        speed=args.robot_speed), pos,
                           radius=args.fov_m)
            active = active_intervals(n_iv, args.explore_ratio)
            print(f"[run_atc]   policy day {di+1}: {n_iv} intervals "
                  f"({int(active.sum())} observed, ratio {args.explore_ratio:g}), "
                  f"{len(set(schedule.tolist()))} distinct regions, "
                  f"loop {sim.total:.0f} m")
            total += drive_day(pipeline, sim, pos, day, xb, yb, args.minutes,
                               stream=stream, gods_eye=args.gods_eye,
                               decimate=args.decimate, active_mask=active,
                               decide_dt_s=DECIDE_DT_S, speed=args.robot_speed)
        else:
            total += drive_day(pipeline, sim, pos, day, xb, yb, args.minutes,
                               stream=stream, gods_eye=args.gods_eye,
                               decimate=args.decimate, speed=args.robot_speed,
                               drift_run=drift_run, node_ids=nav_node_ids)

        if args.checkpoint_every > 0 and (di + 1) % args.checkpoint_every == 0:
            write_checkpoint(di + 1, total)

    if drift_run is not None:
        with open(out_dir / "lc_events.json", "w") as f:
            json.dump(drift_run.events, f, indent=1)
        if drift_run.viz_dir is not None:
            t_end = drift_run.events[-1]["t"] if drift_run.events else 0.0
            dump_viz(pipeline, "final", t_end, drift_run.viz_dir, nav_node_ids)
        print(f"[run_atc] {len(drift_run.events)} loop events, "
              f"{len(pipeline.lc_events())} alignment passes")

    ok, err = pipeline.save_flow_state(args.out_state)
    print(f"[run_atc] {total} detections over {len(args.days)} days; "
          f"save_flow_state ok={ok} err={err!r} -> {args.out_state}")
    try:
        st = pipeline.flow_stats()
        print(f"[run_atc] flow stats: cells={st.get('cells')} "
              f"components={st.get('total_components')}")
    except Exception as e:
        print(f"[run_atc] flow_stats: {e}")


if __name__ == "__main__":
    main()
