#!/usr/bin/env bash
# Fast Online CLiFF-map flow baseline on TBD (heading circular-CRPS + MPP).
# Trains on the 10 TBD train scenes and scores the 6 test scenes in the
# online-mod venv. Two regimes:
#   embodied  A) dump the training stream the robot actually saw, then
#             B) online build + score the held-out scenes.
#   fullvis   train on every detection of every train scene, god's-eye.
# The embodied stream keeps only detections within FOV_M of the LOGGED robot
# pose (each scene's poses.csv), which is what KAIROS itself sees: detections
# outside the active voxel window are dropped in processFrame
# (kairos/src/temporal/flow_temporal_module.cpp), and that window is
# map_window.max_radius_m in kairos/config/datasets/tbd.yaml.
# TBD is a real robot recording, so Stage A needs no authored graph and no
# spark_dsg, and runs in the venv rather than in Docker like ATC / HB.
# Published online CLiFF-map grid resolution is 1 m; override with RADIUS/STEP=0.4
# for a resolution-matched comparison against the 0.4 m batch baselines. The
# Stage A stream is resolution-independent, so SKIP_EXTRACT=1 reuses an existing
# EXTRACT_DIR and goes straight to the build.
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA_ROOT:-$HOME/datasets}"
VENV="${VENV:?set VENV to the python interpreter of the baseline method venv}"
EVAL="$SUITE/baselines/rescore"
EXTRACT_DIR="${EXTRACT_DIR:-$SUITE/results/online_cliff/tbd_extract}"
OUT="${OUT:-$SUITE/results/online_cliff/heading_dist_tbd.txt}"
MINUTES="${MINUTES:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
REGIME="${REGIME:-embodied}"   # embodied | fullvis
# Matches map_window.max_radius_m in kairos/config/datasets/tbd.yaml, and the
# fov_m the ATC / HB extracts record in their bounds.json.
FOV_M="${FOV_M:-5.0}"
EXTRA=""
[[ -n "${RADIUS:-}" ]] && EXTRA="$EXTRA --radius $RADIUS"
[[ -n "${STEP:-}" ]] && EXTRA="$EXTRA --step $STEP"
# Persist the fitted map so a metric change is a re-score, not a retrain.
[[ -n "${WORK_DIR:-}" ]] && EXTRA="$EXTRA --work-dir $WORK_DIR"

# Train = Dec 2022 + Jan 2023 (10 scenes); test = Feb 2023 (6 scenes).
# One shared world frame, so pooling scenes is valid.
T=$DATA/tbd
TRAIN="$T/2022-12/scene_tbd_2022-12-07-12-15-36 $T/2022-12/scene_tbd_2022-12-07-12-44-46 \
$T/2022-12/scene_tbd_2022-12-07-13-21-55 $T/2022-12/scene_tbd_2022-12-07-13-46-01 \
$T/2023-01/scene_tbd_2023-01-23-13-15-54 $T/2023-01/scene_tbd_2023-01-23-13-52-34 \
$T/2023-01/scene_tbd_2023-01-23-14-16-02 $T/2023-01/scene_tbd_2023-01-26-12-54-28 \
$T/2023-01/scene_tbd_2023-01-26-13-36-59 $T/2023-01/scene_tbd_2023-01-26-15-17-19"
TEST="$T/2023-02/scene_tbd_2023-02-01-12-48-21 $T/2023-02/scene_tbd_2023-02-01-13-02-13 \
$T/2023-02/scene_tbd_2023-02-01-13-51-05 $T/2023-02/scene_tbd_2023-02-02-12-05-25 \
$T/2023-02/scene_tbd_2023-02-02-12-14-22 $T/2023-02/scene_tbd_2023-02-03-11-42-50"

if [[ "$REGIME" == fullvis ]]; then
  # God's-eye: the self-contained scorer with its own scene lists, unchanged, so
  # the published full-visibility row stays reproducible.
  echo "[tbd] Stage B (full visibility): 10 train scenes -> $OUT"
  "$VENV" -u "$EVAL/rescore_heading_dist_online_cliff_tbd.py" --out "$OUT" $EXTRA
else
  mkdir -p "$EXTRACT_DIR" "$(dirname "$OUT")"
  if [[ "$SKIP_EXTRACT" == 1 ]]; then
    echo "[tbd] Stage A: skipped, reusing $EXTRACT_DIR"
  else
    echo "[tbd] Stage A: extract FoV training stream (10 scenes, ${FOV_M} m) -> $EXTRACT_DIR"
    "$VENV" -u "$EVAL/extract_fov_stream_online_cliff.py" \
      --stream-source tbd --days $TRAIN --out-dir "$EXTRACT_DIR" \
      --fov-m "$FOV_M" --minutes "$MINUTES"
  fi
  echo "[tbd] Stage B (embodied): online build + score 6 test scenes -> $OUT"
  "$VENV" -u "$EVAL/rescore_heading_dist_online_cliff_embodied.py" \
    --extract-dir "$EXTRACT_DIR" --test-days $TEST --out "$OUT" $EXTRA
fi
echo "[tbd] done -> $OUT"
