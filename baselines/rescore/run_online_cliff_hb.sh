#!/usr/bin/env bash
# Fast Online CLiFF-map flow baseline on Havlickuv Brod (HB); heading
# circular-CRPS + MPP). Same two-stage protocol and train/test split as the
# aion HB baseline: 227 train dates (2022-09-05..2023-04-23) + 28 test dates
# (2023-04-24..2023-05-21), all-camera dates only.
#   A) Docker (kairos-suite): dump the robot field-of-view TRAINING stream.
#   B) venv: online build + score the held-out test days.
# WARNING: the full run is multi-hour. Set MINUTES for a quick slice.
# Published online CLiFF-map grid resolution is 1 m; override with RADIUS/STEP=0.4
# for a resolution-matched comparison against the 0.4 m batch baselines. The
# Stage A stream is resolution-independent, so SKIP_EXTRACT=1 reuses an existing
# EXTRACT_DIR and goes straight to the build.
# Dataset location: set GRAPH, HB_ROOT (container path) and HB_HOST_ROOT (the
# same data from the host) if yours is not under $DATA_ROOT/hb-station.
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA_ROOT:-$HOME/datasets}"
IMAGE="${IMAGE:-kairos-suite}"
VENV="${VENV:?set VENV to the python interpreter of the baseline method venv}"
EXTRACT_DIR="${EXTRACT_DIR:-$SUITE/results/online_cliff/hb_extract}"
OUT="${OUT:-$SUITE/results/online_cliff/heading_dist_hb.txt}"
MINUTES="${MINUTES:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
REGIME="${REGIME:-embodied}"   # embodied | fullvis
# Dataset location. HB_ROOT is the container path; HB_HOST_ROOT is the same
# data as seen from the host, for the date listing below.
GRAPH="${GRAPH:-/data/hb-station/graph/hb_flow_graph.json}"
HB_ROOT="${HB_ROOT:-/data/hb-station/Data-2}"
HB_HOST_ROOT="${HB_HOST_ROOT:-$DATA/hb-station/Data-2}"
EXTRA=""
[[ -n "${RADIUS:-}" ]] && EXTRA="$EXTRA --radius $RADIUS"
[[ -n "${STEP:-}" ]] && EXTRA="$EXTRA --step $STEP"
# Persist the fitted map so a metric change is a re-score, not a retrain.
[[ -n "${WORK_DIR:-}" ]] && EXTRA="$EXTRA --work-dir $WORK_DIR"

# Test window (28 dates) and train (all earlier all-camera HB dates).
TEST_DATES="2023-04-24 2023-04-25 2023-04-26 2023-04-27 2023-04-28 2023-04-29 2023-04-30 \
2023-05-01 2023-05-02 2023-05-03 2023-05-04 2023-05-05 2023-05-06 2023-05-07 2023-05-08 \
2023-05-09 2023-05-10 2023-05-11 2023-05-12 2023-05-13 2023-05-14 2023-05-15 2023-05-16 \
2023-05-17 2023-05-18 2023-05-19 2023-05-20 2023-05-21"
mkdir -p "$EXTRACT_DIR" "$(dirname "$OUT")"

if [[ "$SKIP_EXTRACT" == 1 ]]; then
  echo "[hb] Stage A: skipped, reusing $EXTRACT_DIR"
else
  mapfile -t TRAIN < <(TEST="$TEST_DATES" SIMDIR="$SUITE/grounded_sim" HB_HOST_ROOT="$HB_HOST_ROOT" "$VENV" - <<'PY'
import os,sys
sys.path.insert(0, os.environ["SIMDIR"])
from hb_stream import station_dates
test=set(os.environ["TEST"].split())
dates=sorted(station_dates(os.environ["HB_HOST_ROOT"],"HB").keys())
print("\n".join(d for d in dates if d not in test))
PY
)
  echo "[hb] Stage A: extract FoV training stream (${#TRAIN[@]} dates) -> $EXTRACT_DIR"
  docker run --rm --name online_cliff_hb_extract \
    -v "$SUITE":/ws/src/kairos-suite \
    -v "$DATA":/data:ro -v "$EXTRACT_DIR":/out \
    "$IMAGE" bash -c "set +u; source /opt/ros/noetic/setup.bash; source /ws/devel/setup.bash; export PYTHONPATH=/ws/src/kairos/python/src:\$PYTHONPATH; set -u; \
      python3 -u /ws/src/kairos-suite/baselines/rescore/extract_fov_stream_online_cliff.py \
        --graph $GRAPH --stream-source hb --hb-root $HB_ROOT --station HB \
        --days ${TRAIN[*]} --out-dir /out --minutes $MINUTES; \
      chown -R $(id -u):$(id -g) /out"
fi

if [[ "$REGIME" == fullvis ]]; then
  # Full visibility: train on every detection in the scored rectangle, streamed
  # one day at a time. The rectangle is the one every other HB row scores.
  mapfile -t TRAIN < <(TEST="$TEST_DATES" SIMDIR="$SUITE/grounded_sim" HB_HOST_ROOT="$HB_HOST_ROOT" "$VENV" - <<'PYFV'
import os,sys
sys.path.insert(0, os.environ["SIMDIR"])
from hb_stream import station_dates
test=set(os.environ["TEST"].split())
dates=sorted(station_dates(os.environ["HB_HOST_ROOT"],"HB").keys())
print("\n".join(d for d in dates if d not in test))
PYFV
)
  echo "[hb] Stage B (full visibility): ${#TRAIN[@]} train days -> $OUT"
  "$VENV" -u "$SUITE/baselines/rescore/rescore_heading_dist_online_cliff_fullvis.py" \
    --source hb --bounds "$EXTRACT_DIR/bounds.json" \
    --train-days "${TRAIN[@]}" --test-days $TEST_DATES --out "$OUT" $EXTRA
else
  echo "[hb] Stage B (embodied): online build + score test days -> $OUT"
  "$VENV" -u "$SUITE/baselines/rescore/rescore_heading_dist_online_cliff_embodied.py" \
    --extract-dir "$EXTRACT_DIR" --test-days $TEST_DATES --out "$OUT" $EXTRA
fi
echo "[hb] done -> $OUT"
