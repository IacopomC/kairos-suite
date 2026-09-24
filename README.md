# Kairos evaluation suite

Trains and evaluates flow models over recorded pedestrian datasets: Kairos, and
any method with an adapter in [`baselines/`](baselines/).

The Kairos model itself is built in the [`kairos`](https://github.com/IacopomC/kairos) repository.

| Package | What it does |
|---|---|
| [`grounded_sim/`](grounded_sim/) | Author a 3DSG over a floor map, drive a simulated robot with a bounded field of view over a real pedestrian stream, and train a flow model from what it sees. ATC and Havlíčkův Brod (HB). Optional Rerun recording. |
| [`evaluation/`](evaluation/) | Stream held-out detections through a trained model and accumulate every metric in one pass: flow, presence, calibration, coverage stratification, coupling. |
| [`baselines/`](baselines/) | One adapter per method, plus the contract for adding your own. Kairos is the reference adapter. |
| [`people_finding/`](people_finding/) | One fixed planner over an episode set, evaluated against a ground-truth encounter replay. |
| [`timing/`](timing/) | Pools per-scene flow timing into the cost numbers. |
| [`dataset-analysis/`](dataset-analysis/) | Measures the data itself: is there recurrent structure here, how much of the error is tracking noise.  |

## Installation

The suite runs on top of Kairos, in Docker or natively (Ubuntu 20.04, ROS
Noetic and a native Kairos build).

### 1. Set up the workspace

The Kairos workspace already holds this repository as `kairos-suite`:

```bash
mkdir -p ~/kairos_ws/src && cd ~/kairos_ws/src
git clone https://github.com/IacopomC/kairos.git
vcs import . < kairos/install/kairos.rosinstall   # needs python3-vcstool
```

### 2a. Docker

The `kairos-suite` image is the self-contained Kairos image plus this
repository and its plotting dependencies:

```bash
cd ~/kairos_ws/src
kairos/docker/build_image.sh                                        # -> kairos:latest
cd kairos-suite && docker build -t kairos-suite -f docker/Dockerfile .  # -> kairos-suite
```

Every script takes `IMAGE=` to override the tag; it defaults to `kairos-suite`.
The scripts mount this checkout into the image, so edits to the suite apply
without a rebuild; a change to Kairos itself needs both images rebuilt.
`docker/run.sh <command>` runs any suite command in the image with the data and
results folders mounted, and `docker/run.sh bash` opens a shell there.

### 2b. Native

Build Kairos natively first (section *2b. Native* of the
[`kairos` README](https://github.com/IacopomC/kairos#2b-native-ubuntu-2004--ros-noetic)),
then add the suite's Python dependencies and put its packages on the Python
path:

```bash
pip install "matplotlib==3.7.5" "pandas==2.0.3" pyyaml msgpack
export PYTHONPATH=~/kairos_ws/src/kairos-suite:~/kairos_ws/src/kairos-suite/evaluation:~/kairos_ws/src/kairos-suite/grounded_sim:$PYTHONPATH
```

The end-to-end scripts below run in Docker; natively, run their stages
directly (`grounded_sim/train_atc.py`, then `evaluation/evaluate.py`, as in
[`grounded_sim/README.md`](grounded_sim/README.md)).

### Data

Datasets live under `$DATA_ROOT` (default `$HOME/datasets`). The per-dataset layout is in
[`grounded_sim/README.md`](grounded_sim/README.md) (ATC, HB) and in
[`evaluation/run_tbd.sh`](evaluation/run_tbd.sh) (TBD). Results land in
`results/` of this checkout.

## Training and evaluating a model

Each script trains a flow model on a dataset's training days, then evaluates
it online on the held-out days, in the `kairos-suite` image:

```bash
export DATA_ROOT=/path/to/datasets

DATASET=atc grounded_sim/train_test.sh    # -> results/atc/
DATASET=hb  grounded_sim/train_test.sh    # -> results/hb/
evaluation/run_tbd.sh                     # -> results/tbd/
docker logs -f kairos-atc                 # (or kairos-hb, kairos-tbd) to follow a run
```

ATC and HB are grounded simulations: a simulated robot patrols the authored
navigation graph with a 5 m field of view and trains on what it sees. TBD runs
`kairos run` on the real recordings: the training sessions in order, each
resuming the previous flow state, then every test session evaluated against the
final state, pooled into one report. Each script has a default train/test
split; `TRAIN_DAYS` / `TEST_DAYS` (ATC, HB) and `TRAIN_SCENES` / `TEST_SCENES`
(TBD) choose another one.

A run writes the trained model (`train/flow_state.bin`) and the evaluation
report `metrics_online.txt`: coverage (`match_rate`), MLPD of heading, speed
and the joint, circular CRPS, speed MAE, calibration, coverage strata and
coupling. ATC and HB also write `presence_online.txt`, the presence forecast
evaluated at horizons from 5 s to 600 s.

## Evaluating a different method

`--method` names an adapter in [`baselines/`](baselines/):

```bash
docker/run.sh python3 evaluation/evaluate.py --method kairos --dataset atc --days <test npz...> ...
docker/run.sh python3 evaluation/evaluate.py --method stef   --dataset atc --days <test npz...> \
    --stefmap-path /path/to/stefmap --stef-train-days <train npz...>
```

Adding your own is one file: copy `baselines/template_adapter.py` to
`baselines/<name>_adapter.py`, implement `make_scorer`, and run with
`--method <name>`. The contract is documented in
[`baselines/README.md`](baselines/README.md).

The baseline methods themselves live in their own repositories. Check them out
separately and pass their location (`--stefmap-path` and equivalents).

## Tools

| Output | Produced by |
|---|---|
| Flow metrics, calibration, coverage strata, coupling (ATC, HB) | `evaluation/evaluate.py`, driven by `grounded_sim/train_test.sh` |
| Flow metrics (TBD) | `kairos run --score-tracks` per test session, pooled by `evaluation/score_preds.py`, driven by `evaluation/run_tbd.sh` |
| Presence (ATC; HB with `--stream-source hb`) | `evaluation/evaluate.py --presence` |
| Sparse-coverage statistics and heatmaps | `evaluation/compute_sparse_coverage_stats.py`, `evaluation/plot_sparse_coverage.py` |
| People-finding evaluation | `people_finding/run_finding.py` |
| Cost and timing | `timing/aggregate_flow_timing.py` |

`kairos run --score-tracks` locates `evaluation/score_helper.py` on its own, or
set `KAIROS_SUITE_DIR` to the `evaluation/` directory.

## Visualization

A run can emit a portable `.npz` recording that renders to a scrubbable Rerun
`.rrd`, rendered in the small `kairos-viz` image. `grounded_sim/create_rrd.sh` builds it from the
`kairos` checkout next to this one (`KAIROS_DIR=` to point elsewhere), runs the
simulation, and renders:

```bash
STREAM_SOURCE=atc GRAPH=/data/atc/graph/atc_graph.json \
  DAYS="/data/atc/npz/atc-20121114.npz" grounded_sim/create_rrd.sh   # -> results/viz/out.rrd
rerun results/viz/out.rrd                                            # pip install rerun-sdk
```

`STREAM_SOURCE={atc,hb}` picks the dataset. `MODE=online` builds the model with the roaming robot, so coverage and arrows evolve; `MODE=predict` forecasts from a frozen `FLOW_STATE` over held-out days with the real detections overlaid.

## Bringing in a new dataset

[`dataset-analysis/periodicity/`](dataset-analysis/) measures whether walking
direction in a dataset has a recurrent rhythm, against a split-half noise floor
built from even and odd days, and how much of the heading error is tracking
noise. See [`dataset-analysis/README.md`](dataset-analysis/README.md).
