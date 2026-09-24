# kairos-suite — Metrics Reference

What each metric reported by `evaluate.py` measures, why it was chosen, and how
to read it. For background on the system, see `Concepts.md`.

---

## Coverage-aware MLPD — `mlpd_heading`, `mlpd_speed`, `mlpd_joint`

**What it tests:** the quality of the predicted flow distribution at held-out
detections, per channel: heading `p(θ)`, speed `p(ρ)` and joint `p(θ, ρ)`.

**Formula:** `mean over all test detections of log p(θ_obs, ρ_obs | model at t)`,
with the density evaluated at each detection's own time.

**Plain language:** for each detection the model predicts a distribution over
headings and speeds; MLPD is the log-density it assigned to what was actually
observed. A tall, correctly placed peak does well; a flat guess does
consistently and modestly; a confident peak in the wrong direction does very
poorly. More negative is worse.

**Why a density metric.** The mixture is multimodal (a corridor with two-way
traffic puts mass on two opposite headings). MLPD evaluates the full distribution,
so both modes count; a point error such as heading MAE collapses them into one
"dominant heading", which is wrong whenever two directions are equally likely.

**Coverage.** MLPD is averaged over *all* held-out detections. A detection in a
voxel outside the model's coverage is unmatched and receives the uniform-density
floor over `[0, 2π) × [0, ρ_max]` (`-log 2π` for heading, `-log ρ_max` for
speed, their sum for the joint), so MLPD charges a model for every detection it
leaves uncovered. `ρ_max` is the physical speed cap (`--rho-max`, or
the capped running maximum), identical for every method. Matched densities are
clamped below at `1e-9`. `match_rate` (the fraction of covered detections) and
the matched-only MLPD are reported alongside, as is the fraction of matched
detections below the density floor.

---

## Speed MAE — `speed_mae_mps`

**What it tests:** the error of the predicted speed, in m/s, over matched
detections. Each slot's speed is unimodal, so a point error is meaningful here.
Its detection subsets therefore differ across methods.

---

## Circular CRPS and MPP — heading distribution

**Circular CRPS (rad, lower is better).** Evaluates the whole predicted heading
distribution by its arc distance to the observed heading:
`CRPS(F, θ) = E_F d(Θ, θ) − ½ E_F d(Θ, Θ')`, with `d` the arc distance. It is
distance-aware: mass placed 30° from the observation costs less than
mass placed 180° away. A uniform forecast gets `π/4`, which is also the floor
for unmatched detections.

**MPP (higher is better).** The probability mass on the observed 45° sector
(uniform = 0.125). It rewards concentrating mass on the modal sector, which makes
it an improper score; it is reported as an interpretable secondary.

Three further views of the same metrics are printed:

- **Binned mixture.** The same forecast integrated over the shared 45° sectors
  before evaluation. If binned and continuous values agree, the sector-level
  weights carry the result; if the binned value is better, the within-sector
  kernel shape is the cost.
- **Per passage.** One vote per (track, voxel) crossing, so a dweller or a
  slow walker weighs in the mean once per crossing.
- **Per cell.** Every occupied cell contributes its own mean once
  (`--cellmap-out` writes the per-cell arrays), so rarely observed cells count
  as much as busy ones. 

---

## Speed calibration — `speed_coverage_68`, `speed_coverage_95`, `speed_ece`, `speed_sharpness_mps`

**What it tests:** whether the predicted speed intervals are trustworthy.

- `speed_coverage_68` / `speed_coverage_95`: fraction of observations inside the
  predicted 68 % / 95 % interval (ideal 0.68 / 0.95).
- `speed_ece`: mean gap between predicted and empirical coverage over 20 levels
  (ideal 0).
- `speed_sharpness_mps`: mean interval width `2·std_rho`. Smaller is better, but
  only once coverage is acceptable.

Coverage below nominal means over-confident intervals; above nominal,
conservative ones. The per-slot speed is a running mean, so its interval is the
slot's measured speed scatter blended with the prior spread (see
`speed_scatter_prior_obs` in the Kairos parameter reference).

---

## Weight calibration — `evaluate.py --weight-calibration`

**What it tests:** whether the uncertainty on the **mixing weights** is
trustworthy, evaluated on an observable that the weights predict, since the
weight itself is latent.

**The observable.** A crossing's soft slot responsibilities are averaged per
crossing using the trainer's own visit rules (new visit on voxel change, on
silence longer than `visit_gap_s`, or on a span longer than `visit_max_s`; the
vote is stamped at the visit midpoint; id-less detections vote individually).
The statistic is the mean vote over a non-overlapping window of width
`--weight-dt-win` at one voxel, per slot.

**The predictive.** Mean: the window average of the predicted weight at the
votes' times. Its parameter variance is the weight predictors' posterior,
propagated through the normalization at each slot's effective order and
moment-matched into a Beta (capped below the Bernoulli bound); a binomial layer
adds the scatter of a finite window. Each sample's credible mass is the seeded
randomized PIT of the observed share under this beta-binomial; coverage, ECE and
sharpness follow as for speed.

**Two rows per run:**
- `soft` — the primary variant (shares of soft responsibilities).
- `hard` — argmax-assigned integer counts. Differs from `soft` only through the
  soft-vote layer, so their agreement bounds that layer's conservatism.

**Precision instruments:**
- `bootstrap` — percentile intervals from resampling (voxel, day) clusters, the
  precision that accounts for shared pedestrians and per-voxel dependence.
- `ece_null` — the ECE a *perfect* model would show on this sample structure;
  read the measured ECE against it.

**Strata:** `count` (votes per window), `evidence` (training crossings of the
cell, with the parameter-variance share per bin), `slot`, `clamp`, `cap`,
`order`, `burst`, and a voxel-balanced reweighting.

**Requirements:** `--method kairos`; `--weight-visit-gap-s` /
`--weight-visit-max-s` equal to the config's visit rules; `--weight-dt-win` and
`--weight-min-count` are dataset-scale parameters.

---

## Coverage-stratified MLPD

Joint MLPD reported separately for sparse, medium and dense voxels, by the
number of held-out detections a voxel receives (`--coverage-thresholds LO HI`;
`10 50` on TBD, `5000 20000` on ATC). Evidence sharing lends a voxel its
neighbourhood's evidence with a weight that falls as its own crossings
accumulate, so its effect concentrates in the sparse stratum and vanishes in
the dense one. A whole-dataset average is dominated by the dense stratum.

---

## Phase-stratified and elapsed-time MLPD

- **By phase** (`--temporal-phases name:period_s:n_bins ...`): joint MLPD folded
  by phase within each period, e.g. hour of day (`daily:86400:24`) and day of
  week (`weekly:604800:7`).
- **By time since session start** (`--temporal-elapsed-bins-min`): for short
  sessions such as TBD.

---

## Coupling — `true_j_mlpd_gain`, `cond_heading_mae_gain_deg`

**What it tests:** whether the learned coupling between adjacent voxels carries
information beyond the per-voxel marginals.

**Joint-MLPD gain.** For every pair of detections in adjacent coupled voxels
within `--coupling-window-s`, the log-density of the pair under the coupled
joint, minus the same pair evaluated as two independent marginals. Positive means
the coupling helps.

**Conditional-heading MAE gain.** The heading at one voxel predicted from the
simultaneous observation at its neighbour through the coupling, against the
neighbour's own marginal alone; the gain is the reduction in angular error, in
both directions of each pair.

**Strata:** by `D_KL(π_i ‖ π_j)` between the two voxels' training-time weights
(`low` < 0.2 ≤ `medium` < 1.0 ≤ `high`), by same or different track id, and on
pairs whose observed headings disagree by more than 45°. The coupling state is
exported by the model (`pipeline.flow_coupling_json`), and the metric runs for
methods that export one.

---

## Loop-closure ΔMLPD — `lc_mean_delta_mlpd`, `lc_frac_improved`

Joint MLPD in the window before against the window after each loop-closure
event (`--lc-events-json`, `--lc-window-s`). A positive delta means the
correction improved prediction; `lc_frac_improved` is the fraction of events
with a positive delta. It needs real accumulated drift: on a compact path with
no nearby detections around the events it is empty.

---

## Presence — `evaluate.py --presence`

**What it tests:** the forecast probability that a cell is occupied within a
horizon `Δt` (`--presence-horizons`), evaluated on the binary event it predicts.
Every mapped cell is queried on a time grid across the held-out days (training
stays bounded to the robot's field of view) against the occupancy realised by
all detections.

**Metrics**, per horizon:
- **Bernoulli log score**: strictly proper, dominated by confident misses on the
  rare occupied cells. The primary presence metric.
- **Reliability** (calibration error: how far observed frequencies in each
  forecast-probability bin depart from the probability claimed; 0 is perfect)
  and **resolution** (how far those frequencies spread around the base rate,
  the information the forecast carries; 0 for a constant forecast), from the
  Murphy decomposition of the Brier score.

At low occupancy a single scalar barely separates forecasts, because the empty
majority dominates it; resolution and reliability carry the comparison.

**Estimators:**
- **Kairos**: the forecast occupancy with the dwell correction.
- **Constant base rate**: the training occupancy rate per horizon, derived from
  the FreMEn training pass or supplied with `--base-rate-5s` / `--base-rate-10s`
  (otherwise the in-sample rate, marked `base_rate*` in the report).
  Calibrated and uniform across cells.
- **FreMEn** (`--fremen-train-days`): a per-cell spectral occupancy model fit on
  the training days. It divides occupied time by elapsed time, so unobserved
  time counts as empty time, and it is fit at full visibility.

```
python evaluate.py --method kairos --dataset atc --config-name atc --presence \
    --graph <atc_graph.json> --scene-dir <stub> --flow-state <flow_state.bin> \
    --days <test npz...> --fremen-train-days <train npz...>
```

---

## Runtime and memory — `.stats.json` sidecars

Written by `run_stats.record_stats`, at the end of a training run
(`train.stats.json`) and of an evaluation run (`preds.stats.json`):

- `flow_timing`: each temporal-layer timer at its natural cadence, as
  `<component>_count`, `<component>_total_us` and `<component>_mean_ms` —
  per frame (`process_frame`, `ingest`, `coupling`), per backend event
  (`place_summaries`, `remap_cells`, `run_maintenance`), per query (`score`),
  per detection (`add_observation`), and per pooled neighbourhood (`sharing`,
  counted each time the weights are read).
- `memory`: `rss_bytes`, `peak_rss_bytes`, `vms_bytes`.
- `flow_stats`: cell count, components, valid predictors.
- `stability`: time-to-stability records (observations until a cell's dominant
  slot stops changing), a training-time quantity: it lives outside
  `flow_state.bin`, so it is meaningful only in the training sidecar.

`add_observation` accrues during the training push loop and is null in an
evaluation sidecar. `sharing` accrues wherever the weights are read, so both
sidecars carry it.
