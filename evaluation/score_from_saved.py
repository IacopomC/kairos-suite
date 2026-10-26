#!/usr/bin/env python3
r"""
score_from_saved.py — load a trained flow_state.bin and score test detections.

The piece of the pipeline that the persistence layer (pipeline.save_flow_state
/ pipeline.load_flow_state) was built to enable: train once, save the binary,
then in a fresh process load that binary and score it against any test
scene without re-running the training pass.

What this script does, in order:
    1. Construct a KairosPipeline via the same load_pipeline() path the
       training CLI uses, so config resolution + module wiring is identical.
    2. Call pipeline.load_flow_state(<flow_state.bin>) to restore the full
       per-cell NUDFT / Poisson / timestamps state plus the voxel grid resolution.
    3. Install the authored navigational layer (--graph), the neighbourhoods
       evidence sharing reads.
    4. Run score_helper.score_csv against the test tracks CSV, writing a
       per-detection predictions CSV in the schema evaluate.py consumes.
    5. Run run_stats.record_stats to drop a `<preds>.stats.json` sidecar
       carrying the runtime/memory snapshot evaluate.py will fold into its
       output.

What this script does NOT do:
    - pipeline.start() / pipeline.step() — there's no frame loop. Sensor
      data isn't ingested. The active window, frontend, backend, and
      loop-closure modules sit idle.
    - pipeline.save() — scoring is read-only; we don't want to overwrite
      the trained flow_state.bin with the post-scoring (identical) state,
      and we don't want a fresh DSG write either.

A --scene-dir is still required because KairosPipeline.from_config wants a
sensor object to instantiate the reconstruction module. The dataloader
that scene-dir produces is constructed but never iterated. Using the test
scene's own dir is the natural choice.

Usage:
    python3 score_from_saved.py \
        --scene-dir path/to/test_scene \
        --config-name atc \
        --labelspace ade20k_outdoor \
        --flow-state path/to/train_run/temporal/flow_state.bin \
        --graph path/to/authored_graph \
        --test-tracks path/to/test_scene/tracks.csv \
        --output path/to/preds/atc/preds.csv \
        --label Kairos
"""

import argparse
import os
import pathlib
import sys

import hydra_python

from scene_loader import get_dataloader
from kairos_python.pipeline import load_pipeline  # flow pipeline moved to kairos

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_helper import score_csv  # noqa: E402
from run_stats import record_stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained flow_state.bin and score it against a test tracks CSV. "
            "Pure read-only: no frame ingestion, no fresh DSG save."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        type=str,
        help=(
            "FileDataLoader-format scene directory. Used only to construct "
            "the dataloader/sensor that KairosPipeline.from_config requires; "
            "no frames are iterated. Use the test scene's dir."
        ),
    )
    parser.add_argument(
        "--config-name",
        required=True,
        help="Config name (atc, tbd or hb). Must match the one used at training time.",
    )
    parser.add_argument(
        "--labelspace",
        default="ade20k_outdoor",
        help="Labelspace name (must match training).",
    )
    parser.add_argument(
        "--flow-state",
        required=True,
        type=str,
        help="Path to the flow_state.bin saved at end of training.",
    )
    parser.add_argument(
        "--graph",
        required=True,
        type=str,
        help=(
            "The authored graph the model was trained on. Its navigational "
            "layer is what evidence sharing reads. A real-sensor dataset (TBD) "
            "builds its own scene graph: score it inside `kairos run <test "
            "scene> --load-flow-state <state> --score-tracks <tracks.csv>`."
        ),
    )
    parser.add_argument(
        "--test-tracks",
        required=True,
        type=str,
        help="Path to the test scene's tracks.csv.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output predictions CSV. The matching .stats.json sidecar is written automatically.",
    )
    parser.add_argument(
        "--label",
        required=True,
        type=str,
        help="System label (e.g. Kairos). Stored in the .stats.json sidecar.",
    )
    parser.add_argument(
        "--enable-lcd",
        action="store_true",
        help=(
            "Construct the loop-closure module too. Default off because LCD is "
            "irrelevant for read-only scoring; turning it off also skips the "
            "BoW queue allocation."
        ),
    )
    parser.add_argument(
        "--glog-level",
        default=0,
        type=int,
        help="glog minimum level.",
    )
    parser.add_argument(
        "--verbosity",
        default=0,
        type=int,
        help="glog verbosity.",
    )
    parser.add_argument(
        "--horizon-s",
        default=0.0,
        type=float,
        help=(
            "delta_t passed into cell.snapshot at scoring time. 0.0 = score at "
            "the detection's exact timestamp (the default trajectory-anchored "
            "protocol). Set > 0 for a forward-prediction-horizon experiment."
        ),
    )
    parser.add_argument(
        "--holdout-frac",
        default=0.0,
        type=float,
        help=(
            "When > 0, score only the last `holdout_frac` of the [min_t_ns, "
            "max_t_ns] range in --test-tracks (deterministic time-split). "
            "Use 0.2 for an 80/20 same-env evaluation. For the held-out-week "
            "protocol, pass the held-out CSV as --test-tracks and leave this "
            "at 0.0."
        ),
    )
    args = parser.parse_args()

    hydra_python.set_glog_level(args.glog_level, args.verbosity)

    scene_path = pathlib.Path(args.scene_dir).expanduser().absolute()
    if not scene_path.exists():
        print(f"ERROR: scene dir not found: {scene_path}", file=sys.stderr)
        sys.exit(1)
    if not pathlib.Path(args.flow_state).expanduser().exists():
        print(f"ERROR: flow_state.bin not found: {args.flow_state}", file=sys.stderr)
        sys.exit(1)
    if not pathlib.Path(args.test_tracks).expanduser().exists():
        print(f"ERROR: test tracks not found: {args.test_tracks}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("KAIROS score_from_saved")
    print("=" * 60)
    print(f"scene-dir:    {scene_path}")
    print(f"config-name:  {args.config_name}")
    print(f"labelspace:   {args.labelspace}")
    print(f"flow-state:   {args.flow_state}")
    print(f"graph:        {args.graph}")
    print(f"test-tracks:  {args.test_tracks}")
    print(f"output:       {args.output}")
    print(f"label:        {args.label}")

    # Build the pipeline. load_pipeline is the same construction path the
    # training CLI uses; the only thing we skip is the data iteration loop.
    print("\n[1/4] Constructing pipeline...")
    data = get_dataloader(scene_path)
    pipeline = load_pipeline(
        data,
        args.config_name,
        args.labelspace,
        output_path=None,
        config_verbosity=1,
        place_feature_strategy=None,
        zmq_url=None,
        enable_lcd=args.enable_lcd,
    )
    if pipeline is None:
        print("ERROR: failed to load pipeline", file=sys.stderr)
        sys.exit(1)

    # Restore the trained state. After this, score_observations works
    # immediately because load_flow_state also reconstructs the voxel grid.
    print("\n[2/4] Loading flow state...")
    ok, err = pipeline.load_flow_state(args.flow_state)
    if not ok:
        print(f"ERROR: load_flow_state failed: {err}", file=sys.stderr)
        sys.exit(1)

    flow_stats = pipeline.flow_stats()
    print(
        f"  loaded: {flow_stats.get('cells', '?')} cells, "
        f"{flow_stats.get('total_components', '?')} components total"
    )

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "grounded_sim"))
    from robot_sim import install_nav_layer  # noqa: E402
    n_nodes, n_edges = install_nav_layer(pipeline, args.graph)
    print(f"  nav layer installed: {n_nodes} nodes, {n_edges} edges")

    print("\n[3/4] Scoring test detections...")
    score_stats = score_csv(
        pipeline,
        test_tracks=args.test_tracks,
        output=args.output,
        horizon_s=args.horizon_s,
        holdout_frac=args.holdout_frac,
    )
    print(
        f"  predictions: {args.output}\n"
        f"  matched:     {score_stats['n_matched']}/{score_stats['n_total']} "
        f"({score_stats['match_rate']:.1%})"
        + (
            f"\n  holdout:     last {args.holdout_frac:.0%} of time range "
            f"(skipped {score_stats['n_skipped_holdout']} rows)"
            if args.holdout_frac > 0.0
            else ""
        )
    )

    print("\n[4/4] Recording run stats sidecar...")
    sidecar = record_stats(
        pipeline,
        predictions_csv=args.output,
        label=args.label,
        extra={
            "config_name": args.config_name,
            "flow_state_path": os.path.abspath(args.flow_state),
            "test_tracks_path": os.path.abspath(args.test_tracks),
            "horizon_s": args.horizon_s,
        },
    )
    print(f"  RSS (current):  {(sidecar['memory'].get('rss_bytes') or 0) / 1e6:.1f} MB")
    if sidecar["memory"].get("peak_rss_bytes"):
        print(f"  RSS (peak):     {sidecar['memory']['peak_rss_bytes'] / 1e6:.1f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
