# Grounded Simulation

Runs the full Kairos temporal stack on a real pedestrian dataset with a
simulated robot: the platform patrols the navigation graph authored on the
floor map with a bounded field of view, and the detections it sees feed the
flow model. 
Two datasets are set up here, **ATC** and **Havlíčkův Brod (HB)**, each in two regimes: *embodied*, where the patrol sees only what falls in its 5 m field of view, and *full visibility*, where the model is trained on every detection (`GODS_EYE=1`). Drift and loop closure are injected with `--drift-config` (see `../config/drift/`); odometry is exact unless a drift file is passed.

## Data

Everything is read from `$DATA_ROOT`.

**ATC** (from the ATC data release, `<ATC data release>`), under `$DATA_ROOT/atc/`:

- `npz/atc-YYYYMMDD.npz`: one file per recorded day, columns `t,id,x,y,vel,ang`
  (unix s, person id, mm, mm/s, rad). 92 days, Wed + Sun, Oct 2012 – Nov 2013.
- `ATC-map/localization_grid.{pgm,yaml}`: ROS occupancy grid, 0.05 m/px,
  origin [-60, -40, 0] m, same world frame as the tracks.
- `graph/atc_graph.json` (optional): the navigation graph. When absent,
  `train_test.sh` authors it from the training days.

**HB**: the trajectories of Apeltauer et al., *Spatiotemporal Trajectories of
Pedestrian Mobility at the Train Station*, Sci Data 11:1254 (2024), from their
figshare repository. Place the `Data-2` tree under `$DATA_ROOT/hb-station/`:
one directory per camera (`HB-CAM1` … `HB-CAM5`, `PB-CAM1` … `PB-CAM6`), each
holding `<camera>_<date>_trans.csv`. Only dates covered by every camera of the
station are used. `graph/hb_flow_graph.json` under `$DATA_ROOT/hb-station/` is
optional, as for ATC.

`maps/` holds the occupancy grid of each HB concourse, 0.05 m, walls traced
from the floor-plan panels of the dataset paper's Fig. 3, which sit alongside
them. The panels are © the dataset authors, CC BY 4.0.

## Run

Install or build the `kairos-suite` image first (see the suite README). Then, from this folder:

```bash
DATA_ROOT=/path/to/datasets DATASET=atc ./train_test.sh   # docker logs -f kairos-atc
DATA_ROOT=/path/to/datasets DATASET=hb  ./train_test.sh   # docker logs -f kairos-hb
```

The default split:

| Dataset | Train | Test |
|---|---|---|
| ATC | every day in `atc/npz/` except the test days (84) | 2013-09-01, 04, 11, 15, 18, 22, 25, 29 (8) |
| HB | all-camera dates 2022-09-05 – 2023-04-23 (227) | all-camera dates 2023-04-24 – 2023-05-21 (28) |

`TRAIN_DAYS` / `TEST_DAYS` override it (ATC: day stems such as
`atc-20121024`; HB: ISO dates). The run prints the number of days in each set.

The output folder is `results/<config>/` of this repository and contains:

- `metrics_online.txt`: flow metrics over all held-out detections: coverage
  (`match_rate`), MLPD of heading, speed and the joint (the first value of each
  `MLPD` line; the bracketed one covers matched detections only), circular CRPS
  and speed MAE, then calibration, coverage strata and coupling.
- `presence_online.txt`: presence at 5, 10, 60, 300 and 600 s for the base
  rate, FreMEn and Kairos.
- `train/flow_state.bin`: the trained model, `graph/`: the authored graph when
  one was authored, `SPLIT.txt`: the exact split.

To run a single stage by hand:

```bash
python3 train_atc.py --graph <graph.json> --days <train npz...> \
    --scene-dir stub_scene --config-name atc \
    --out-dir /out/atc/train --out-state /out/atc/train/flow_state.bin
python3 ../evaluation/evaluate.py --method kairos --dataset atc --config-name atc \
    --graph <graph.json> --scene-dir stub_scene \
    --flow-state /out/atc/train/flow_state.bin --coverage-thresholds 5000 20000 \
    --days <test npz...>
```

For HB, add `--stream-source hb --station HB --hb-root /data/hb-station/Data-2`
to both and pass ISO dates as `--days`.

Optional Rerun recording (`create_rrd.sh` is dataset-agnostic; ATC shown, HB via `STREAM_SOURCE=hb`, and `MODE=predict` for a frozen-model forecast):

```bash
DATA_ROOT=/path/to/datasets STREAM_SOURCE=atc GRAPH=/data/atc/graph/atc_graph.json \
  DAYS="/data/atc/npz/atc-20121114.npz" ./create_rrd.sh    # -> results/viz/out.rrd
```

## Scripts

- `train_test.sh`: graph, training and online evaluation (flow + presence),
  chained in the `kairos-suite` image.
- `author_graph.py`: floor map walls + training-day coverage → navigation graph,
  with a `*.provenance.json` sidecar of the days and parameters used.
- `hb_coverage_cache.py`: bins the HB training stream per cell into the
  coverage cache `author_graph.py` reads for HB.
- `author_hb_graph.py`: HB coverage cells alone → a map-free graph.
- `train_atc.py`: train over `--days` (ATC day files, or HB dates with
  `--stream-source hb`), save `flow_state.bin`.
- `run_viz.py`: dataset-agnostic Rerun recording driver (`--stream-source
  {atc,hb}`, `--mode {online,predict}`); emits the portable `.npz`.
- `create_rrd.sh`: runs `run_viz.py`, then renders to `.rrd`, in the pipeline
  image on ROS 2 or the `kairos-viz` image on ROS 1; configured by environment
  (see the script header).
- `explore_patrol.py`: the region patrol of the sparse-coverage study.
- `draw_patrol.py`, `viz_graph.py`, `flow_viz.py`, `extract_flow_series.py`: figures.
- `robot_sim.py`, `atc_stream.py`, `hb_stream.py`, `tbd_stream.py`: stream and
  robot helpers used by the above.

## Baselines

Comparison methods are evaluated through the adapters in `../baselines/`, selected
with `evaluate.py --method <name>`. The methods live in their own repositories:
check them out and pass their location (`--stefmap-path` and equivalents). See
`../baselines/README.md`.
