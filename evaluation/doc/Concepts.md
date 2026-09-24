# kairos-suite — System Concepts

Background for `metrics.md`: the components the metrics are computed over.

---

## What Kairos models

Kairos adds a spatio-temporal **flow** layer to a hierarchical 3D Scene Graph.
Per voxel it maintains a distribution over pedestrian motion `(θ, ρ)` —
heading `θ` and speed `ρ` — and predicts how that distribution changes with
time of day.

- **Active voxel set (TSDF).** Detections are filtered to voxels whose TSDF
  integration weight exceeds a threshold, tying flow statistics to
  surface-reconstructed geometry. Flow state is allocated lazily at a voxel's
  first detection.

- **Fixed-component SW-GMM (K=8).** Each voxel's directional distribution is a
  semi-wrapped Gaussian mixture with `K=8` components placed at the cardinal
  headings `μ_θ^(k) = kπ/4`, `k = 0..7`, with fixed isotropic covariance.
  Component **geometry is fixed**; the per-slot mixing weights `π^(k)` and
  per-slot speed means `μ_ρ^(k)` are what evolve. Because slot `k`
  always means heading `kπ/4` in every voxel, slot indices are globally
  comparable — the prerequisite for aggregation and pairwise coupling.

- **NUDFT spectral predictors.** Each scalar trajectory we forecast (each slot's
  mixing weight, the presence rate, each coupling parameter) is tracked by an
  incremental Non-Uniform DFT predictor over a fixed set of candidate periods.
  A predictor stores a DC term `γ_0` (running mean, the stationary fallback
  before enough samples) plus complex coefficients at the candidate
  frequencies, and predicts the value at any query time; the mixing-weight and
  presence predictors pick their order per cell by prequential error. Per-slot
  speeds are running means, constant over query time.

- **Gamma–Poisson presence.** A separate channel forecasts the time-varying
  probability that *any* agent is present, so an empty cell and a cell with
  unobserved flow are modelled separately.

- **Pairwise coupling.** Per-edge coupling tensors over traversable,
  face-sharing voxel pairs promote the per-voxel marginals into a joint
  distribution over adjacent voxels, enabling conditional queries.

- **Evidence sharing.** When the weights are read, each slot's running mean is
  shrunk toward the count-weighted mean of the voxel's traversable
  neighbourhood, with a weight that falls as the voxel's own crossings
  accumulate — a prior for sparsely observed voxels.

---

## How evaluation works

- **Trajectory-anchored, online evaluation.** Held-out detections are evaluated at
  their own `(t, x, y, θ, ρ)`: the model's NUDFT predictors are evaluated at
  the detection's timestamp to form the predicted SW-GMM, and the density is
  read at the observed `(θ, ρ)`. Evaluation is read-only.

- **Frozen train→evaluate handoff.** A run trains on the training scenes,
  serializes `flow_state.bin`, and a fresh process loads it and evaluates the test
  scenes. Quantities outside the serialized state (e.g. `n_obs_to_stable`) reset
  on load, so they are meaningful only within the training process that
  produced them and are reported from training runs only.

- **Coverage-aware evaluation.** MLPD is averaged over *all* held-out detections.
  A detection in a voxel outside the model's coverage is "unmatched" and
  receives a uniform-density floor (`-log(2π)` for heading, `-log(ρ_max)` for
  speed), so MLPD charges a model for every detection it leaves uncovered. The
  `match_rate` (fraction covered) is reported alongside, and the coverage-free
  "matched-only" MLPD as a secondary value. `ρ_max` is the physical speed cap,
  identical for every method.

- **Channels.** Densities are reported per channel: heading `p(θ)`, speed
  `p(ρ)`, and joint `p(θ, ρ)`. Heading-only baselines report the heading
  channel alone.

---

## Where each metric lives

- `evaluate.py` — the online evaluator: every flow metric in one streaming pass,
  and presence with `--presence`.
- `online_metrics.py` — the accumulators behind every metric.
- `score_helper.py` / `score_from_saved.py` — produce per-detection predictions
  CSVs; `score_preds.py` feeds them through the same accumulators.
- `run_stats.py` — runtime/memory sidecar.
- `inspect_flow_state.py` — summarise a saved `flow_state.bin`.

Per-metric definitions and how to read them: `metrics.md`.
