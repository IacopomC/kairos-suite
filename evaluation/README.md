# evaluation — evaluating a trained model

Evaluates a trained flow state, or any baseline behind an adapter in
[`../baselines/`](../baselines/), on held-out detections. Per-metric definitions are in
[`doc/metrics.md`](doc/metrics.md); the modelling concepts they rely on are in
[`doc/Concepts.md`](doc/Concepts.md).

---

## The evaluation protocol

Held-out detections are evaluated at their own timestamps and locations: for each
`(t, x, y, θ, ρ)` the pipeline looks up the voxel containing `(x, y)`, evaluates
that cell's forecast at `t`, and evaluates the observed `(θ, ρ)` under the
predicted mixture (`pipeline.score_observations`).

A trained model is saved with `pipeline.save_flow_state(path)` (schema-versioned;
load aborts on a version mismatch) and restored with
`pipeline.load_flow_state(path)`, including the voxel grid.

---

## What lives in this folder

- `evaluate.py` — *the online evaluator* (dataset-agnostic). Streams the
  held-out detections in one pass and accumulates every metric incrementally.
  `--method` names the adapter to evaluate.
  One pass reports:
  - channel-wise coverage-aware MLPD (heading `p(θ)`, speed `p(ρ)`, joint
    `p(θ,ρ)`, each with its own uniform floor), match rate and speed MAE;
  - circular CRPS and MPP of the heading distribution, the same forecast
    integrated over the shared 45° sectors, and per-passage and per-cell
    (equal weight per cell) aggregations of the heading metrics;
  - speed calibration (68/95 % coverage, ECE, sharpness) and, with
    `--weight-calibration`, mixing-weight calibration;
  - MLPD stratified by per-voxel coverage (`--coverage-thresholds LO HI`:
    `10 50` on TBD, `5000 20000` on ATC), by phase of each period
    (`--temporal-phases`, e.g. hour of day), and by time since session start;
  - the coupling metric: joint-MLPD gain and conditional-heading MAE of the
    learned coupling over independent neighbours, on adjacent-voxel pairs,
    stratified by the divergence between the two voxels' weights;
  - with `--lc-events-json`, the joint-MLPD change around each loop closure.

  `--presence` runs the presence evaluation instead: every mapped cell is
  queried on a time grid across the held-out days and evaluated against realised
  occupancy for each horizon in `--presence-horizons`, with the Bernoulli log
  score and the reliability/resolution decomposition, for Kairos, the constant
  base rate and FreMEn (`--fremen-train-days`).

- `online_metrics.py` — the streaming accumulators for every metric above.
- `hb_presence.py` — the FreMEn occupancy fit for the HB station stream (used by
  `evaluate.py --stream-source hb`), runnable on its own as a base-rate /
  FreMEn read with `--selftest` checks.
- `score_helper.py` — drives `pipeline.score_observations` over a tracks CSV and
  writes a per-detection predictions CSV; used by `score_from_saved.py` and by
  `kairos run --score-tracks`.
- `score_from_saved.py` — loads a `flow_state.bin` into a fresh pipeline,
  installs the authored navigational layer (`--graph`), evaluates a tracks CSV and writes the predictions CSV with its `.stats.json` sidecar.
- `score_preds.py` — feeds such a predictions CSV through the same accumulators
  `evaluate.py` uses, without re-running the model.
- `run_stats.py` — writes the `.stats.json` sidecar: flow-map statistics,
  per-component timing, process memory, stability records, and the
  loop-closure events and coupling state as their own sidecars.
- `inspect_flow_state.py` — prints a summary of a `flow_state.bin` without
  loading it into the C++ pipeline.
- `compute_sparse_coverage_stats.py`, `plot_sparse_coverage.py` — the per-cell
  statistics and maps of the sparse-coverage study, from the `--cellmap-out`
  dumps of `evaluate.py`.
- `tests/test_weight_calibration.py` — self-tests of the weight-calibration
  metric on simulated data.

---

## Workflow at a glance

```
raw pedestrian data
        │
        ▼
detections export (ATC: per-day .npz; HB: station files; TBD: tracks.csv)
        │
        ├──> TRAIN days ──► grounded_sim/train_*.py ──► flow_state.bin
        │                                                   └──► inspect_flow_state.py
        └──> TEST days ───► evaluate.py (loads flow_state, streams the test days)
                                    │
             ┌──────────────────────┴──────────────────────────┐
             ▼ one detection-streaming pass                     ▼ --presence
   flow MLPD, speed MAE, CRPS/MPP,                     Bernoulli log score,
   calibration, stratified MLPD, coupling              reliability / resolution
```

`grounded_sim/train_test.sh` chains training and both evaluation passes for one
config.
