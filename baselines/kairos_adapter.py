r"""Kairos adapter: score a trained Kairos flow state.

This is the reference implementation of the adapter contract described in
``baselines/__init__.py``. A new method's adapter has the same two-function
shape; ``template_adapter.py`` is a skeleton to copy.

Usage:
    evaluate.py --method kairos --config-name atc \
        --scene-dir ../grounded_sim/stub_scene \
        --flow-state OUT/train/flow_state.bin --graph <graph.json> ...
"""
import pathlib
import sys

import numpy as np

from . import Scorer


def add_arguments(ap):
    """Options this adapter needs beyond the evaluator's own.

    None: the evaluator already defines --config-name, --labelspace,
    --flow-state, --scene-dir, --graph and --horizon-s, because the
    grounded-simulation trainer takes the same ones.
    """
    return


def require_graph(args):
    """Stop unless --graph is given; the evaluator calls this before any loading."""
    if args.graph is None:
        raise SystemExit(
            "[evaluate] --method kairos needs --graph, the authored graph the "
            "model was trained on: its navigational layer is what evidence "
            "sharing reads. A real-sensor dataset (TBD) builds its own scene "
            "graph, so score it inside `kairos run <test scene> "
            "--load-flow-state <state> --score-tracks <tracks.csv>` and "
            "aggregate with evaluation/score_preds.py.")


def make_scorer(args, ctx):
    """Load the trained state and wrap its score_fn."""
    require_graph(args)
    score_fn, pipeline = kairos_scorer(
        args.config_name, args.labelspace, args.flow_state, args.scene_dir,
        args.horizon_s,
        nav_graph=args.graph,
        weight_channels=args.weight_calibration)

    coupling_json = None
    if not args.no_coupling:
        if args.coupling_json is not None:
            coupling_json = open(args.coupling_json).read()
        else:
            coupling_json = pipeline.flow_coupling_json()

    # The pipeline is held on the Scorer: the score_fn closes over it, and the
    # trained state is released when the run ends.
    return Scorer(score_fn=score_fn, coupling_json=coupling_json, handle=pipeline)


def kairos_scorer(config_name, labelspace, flow_state, scene_dir, horizon_s,
                  nav_graph=None, weight_channels=False):
    """Load a trained KAIROS flow state; return a (score_fn, teardown) pair.

    `nav_graph` (the authored graph the model was trained on) installs the
    navigational layer before scoring. Evidence sharing reads a voxel's place
    neighbourhood at query time, and a scoring run loads the state onto an empty
    pipeline, so the layer has to be installed for sharing to act. Scoring
    pushes no detections and creates no cells, so one place pass is complete.
    """
    import hydra_python
    from kairos_python.pipeline import load_pipeline
    data = hydra_python.get_dataloader(scene_dir)
    pipeline = load_pipeline(data, config_name, labelspace, output_path=None, enable_lcd=False)
    if pipeline is None:
        raise SystemExit("[evaluate] failed to load pipeline")
    ok, err = pipeline.load_flow_state(flow_state)
    if not ok:
        raise SystemExit(f"[evaluate] load_flow_state failed: {err}")
    st = pipeline.flow_stats()
    print(f"[evaluate] loaded {st.get('cells')} cells, {st.get('total_components')} components")

    if nav_graph is not None:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "grounded_sim"))
        from robot_sim import install_nav_layer
        n_nodes, n_edges = install_nav_layer(pipeline, nav_graph)
        print(f"[evaluate] nav layer installed: {n_nodes} nodes, {n_edges} edges")

    def score(t_s, xyz, theta, rho):
        s = pipeline.score_observations(timestamps_s=t_s, xyz_world=xyz,
                                        theta=theta, rho=rho, horizon_s=horizon_s)
        n = len(t_s)
        out = {
            "matched": np.asarray(s["matched"], dtype=bool),
            "log_p": {"heading": np.asarray(s.get("log_p_heading", [np.nan] * n), float),
                      "speed": np.asarray(s.get("log_p_speed", [np.nan] * n), float),
                      "joint": np.asarray(s["log_p_joint"], float)},
            "mean_rho": np.asarray(s["mean_rho"], float),
            "std_rho": np.asarray(s.get("std_rho", [0.0] * n), float),
            # Per-slot mixing weights π^(k)(t) for the coupling metric (ragged:
            # each row is the cell's component weights, empty when unmatched).
            "slot_weights": s.get("slot_weights", [[] for _ in range(n)]),
        }
        if weight_channels:
            # Weight-calibration channels: the propagated per-slot weight stds
            # (+ clamp / effective-order diagnostics and the cell's training
            # evidence) from the score API, and the observed soft slot
            # responsibilities of the raw (theta, rho) detections -- the same
            # model-independent definition the weight predictors are trained
            # on, evaluated by the pipeline's own C++.
            out["slot_weight_stds"] = s["slot_weight_stds"]
            out["slot_weight_clamped"] = s["slot_weight_clamped"]
            out["slot_weight_orders"] = s["slot_weight_orders"]
            out["n_crossings"] = np.asarray(s["n_crossings"], np.int64)
            out["responsibilities"] = np.asarray(
                pipeline.responsibilities(theta=list(theta), rho=list(rho)), float)
        return out
    return score, pipeline
