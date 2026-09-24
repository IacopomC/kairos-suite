#!/usr/bin/env bash
# Grounded simulation, end to end, in the kairos-suite image: author the
# navigation graph (unless one is given), train the flow model with the
# embodied patrol over the training days, then score the held-out days online
# (flow MLPD / CRPS / speed MAE / coverage strata / coupling), then presence.
#
#   DATA_ROOT=/path/to/datasets DATASET=atc ./train_test.sh
#   DATA_ROOT=/path/to/datasets DATASET=hb  ./train_test.sh
#   docker logs -f kairos-atc                 # follow the run
#
# The default split is the paper's:
#   atc  test = the 8 September 2013 days below, train = every other day in
#        $DATA_ROOT/atc/npz (84 of the 92 recorded days)
#   hb   train = all-camera dates 2022-09-05 .. 2023-04-23 (227),
#        test  = all-camera dates 2023-04-24 .. 2023-05-21 (28)
# Override with TRAIN_DAYS / TEST_DAYS (atc: day stems such as atc-20121024;
# hb: ISO dates).
#
# Data layout under $DATA_ROOT (mounted read-only at /data):
#   atc/npz/atc-YYYYMMDD.npz              one file per recorded day
#   atc/ATC-map/localization_grid.{pgm,yaml}
#   hb-station/Data-2/<camera>/<camera>_<date>_trans.csv
# The navigation graph is read from atc/graph/atc_graph.json or
# hb-station/graph/hb_flow_graph.json when the data folder holds one (or from
# GRAPH, an in-container path); otherwise it is authored from the training days
# into $RESULTS/graph/.
#
# Env: DATASET (atc|hb), CONFIG (default = DATASET), DATA_ROOT, RESULTS,
#   IMAGE, GRAPH, TRAIN_DAYS, TEST_DAYS, COVERAGE_THRESHOLDS, NO_PRESENCE=1,
#   GODS_EYE=1 (full-visibility training), DECIMATE (god's-eye only), FOV_M,
#   MINUTES (0 = whole days), NAME (container name).
set -euo pipefail

SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-atc}"
case "$DATASET" in atc|hb) ;; *) echo "DATASET must be atc or hb"; exit 1;; esac
CONFIG="${CONFIG:-$DATASET}"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"
RESULTS="${RESULTS:-$SUITE/results/$CONFIG}"
IMAGE="${IMAGE:-kairos-suite}"
GRAPH="${GRAPH:-}"
TRAIN_DAYS="${TRAIN_DAYS:-}"
TEST_DAYS="${TEST_DAYS:-}"
COVERAGE_THRESHOLDS="${COVERAGE_THRESHOLDS:-5000 20000}"
NO_PRESENCE="${NO_PRESENCE:-0}"
GODS_EYE="${GODS_EYE:-0}"
DECIMATE="${DECIMATE:-1}"
FOV_M="${FOV_M:-5.0}"        # the CONFIG carries the matching map_window.max_radius_m
MINUTES="${MINUTES:-0}"
NAME="${NAME:-kairos-$CONFIG}"

[[ -d "$DATA_ROOT" ]] || { echo "DATA_ROOT=$DATA_ROOT does not exist"; exit 1; }
mkdir -p "$RESULTS"

read -r -d '' CHAIN <<'EOF' || true
set -euo pipefail
cd /ws/src/kairos-suite/grounded_sim
STUB=/ws/src/kairos-suite/grounded_sim/stub_scene
HB_ROOT=/data/hb-station/Data-2

# ---- split ------------------------------------------------------------------
if [ "$DATASET" = atc ]; then
  ATC_TEST="atc-20130901 atc-20130904 atc-20130911 atc-20130915 atc-20130918 atc-20130922 atc-20130925 atc-20130929"
  [ -n "$TEST_DAYS" ] || TEST_DAYS="$ATC_TEST"
  if [ -z "$TRAIN_DAYS" ]; then
    TRAIN_DAYS=$(python3 - "$TEST_DAYS" <<'PY'
import glob, os, sys
test = set(sys.argv[1].split())
days = sorted(os.path.basename(p)[:-4] for p in glob.glob("/data/atc/npz/atc-*.npz"))
print(" ".join(d for d in days if d not in test))
PY
)
  fi
  TRAIN_ARGS=""; for d in $TRAIN_DAYS; do TRAIN_ARGS="$TRAIN_ARGS /data/atc/npz/$d.npz"; done
  TEST_ARGS="";  for d in $TEST_DAYS;  do TEST_ARGS="$TEST_ARGS /data/atc/npz/$d.npz"; done
  for f in $TRAIN_ARGS $TEST_ARGS; do [ -f "$f" ] || { echo "missing day file $f"; exit 1; }; done
  SRC_ARGS=""
else
  hb_dates() {
    python3 - "$1" "$2" <<'PY'
import sys
from hb_stream import station_dates
lo, hi = sys.argv[1], sys.argv[2]
print(" ".join(d for d in station_dates("/data/hb-station/Data-2", "HB") if lo <= d <= hi))
PY
  }
  [ -n "$TRAIN_DAYS" ] || TRAIN_DAYS=$(hb_dates 2022-09-05 2023-04-23)
  [ -n "$TEST_DAYS" ]  || TEST_DAYS=$(hb_dates 2023-04-24 2023-05-21)
  TRAIN_ARGS="$TRAIN_DAYS"; TEST_ARGS="$TEST_DAYS"
  SRC_ARGS="--stream-source hb --station HB --hb-root $HB_ROOT"
fi
N_TRAIN=$(echo $TRAIN_DAYS | wc -w); N_TEST=$(echo $TEST_DAYS | wc -w)
[ "$N_TRAIN" -gt 0 ] && [ "$N_TEST" -gt 0 ] || { echo "empty split: train=$N_TRAIN test=$N_TEST (check DATA_ROOT layout)"; exit 1; }
echo "[split] $DATASET: $N_TRAIN train days, $N_TEST test days"
{ echo "DATASET=$DATASET"; echo "CONFIG=$CONFIG"; echo "GODS_EYE=$GODS_EYE"; echo "DECIMATE=$DECIMATE";
  echo "FOV_M=$FOV_M"; echo "MINUTES=$MINUTES"; echo "TRAIN_DAYS=$TRAIN_DAYS"; echo "TEST_DAYS=$TEST_DAYS"; } > /out/SPLIT.txt

# ---- graph ------------------------------------------------------------------
if [ -z "$GRAPH" ]; then
  if [ "$DATASET" = atc ]; then SHIPPED=/data/atc/graph/atc_graph.json; else SHIPPED=/data/hb-station/graph/hb_flow_graph.json; fi
  [ -f "$SHIPPED" ] && GRAPH="$SHIPPED"
fi
if [ -z "$GRAPH" ]; then
  mkdir -p /out/graph
  if [ "$DATASET" = atc ]; then
    GRAPH=/out/graph/atc_graph.json
    [ -f "$GRAPH" ] || python3 -u author_graph.py --days $TRAIN_ARGS \
      --map-pgm /data/atc/ATC-map/localization_grid.pgm \
      --map-yaml /data/atc/ATC-map/localization_grid.yaml \
      --out "$GRAPH" --out-occupancy /out/graph/atc_occupancy.npz \
      --coverage-cache /out/graph/coverage_cache.npz
  else
    GRAPH=/out/graph/hb_flow_graph.json
    if [ ! -f "$GRAPH" ]; then
      python3 -u hb_coverage_cache.py --hb-root $HB_ROOT --station HB \
        --map-yaml maps/hb_occupancy.yaml --dates $TRAIN_DAYS \
        --out /out/graph/hb_coverage_cache.npz
      python3 -u author_graph.py --days $TRAIN_DAYS \
        --map-pgm maps/hb_occupancy.pgm --map-yaml maps/hb_occupancy.yaml \
        --coverage-cache /out/graph/hb_coverage_cache.npz --out "$GRAPH"
    fi
  fi
fi
echo "[graph] $GRAPH"

# ---- train ------------------------------------------------------------------
# A trained state is reused only if it was trained on this same split.
SPLIT_KEY="$DATASET|$CONFIG|$GODS_EYE|$DECIMATE|$FOV_M|$MINUTES|$GRAPH|$TRAIN_DAYS"
mkdir -p /out/train
if [ -f /out/train/flow_state.bin ] && [ "$(cat /out/train/split.key 2>/dev/null)" = "$SPLIT_KEY" ]; then
  echo "[train] flow_state.bin for this split exists, skipping training"
else
  rm -f /out/train/flow_state.bin /out/train/split.key
  if [ "$GODS_EYE" = 1 ]; then REGIME="--gods-eye --decimate $DECIMATE"; else REGIME="--fov-m $FOV_M"; fi
  python3 -u train_atc.py $SRC_ARGS --graph "$GRAPH" --days $TRAIN_ARGS --scene-dir $STUB \
    --config-name $CONFIG --out-dir /out/train --out-state /out/train/flow_state.bin \
    --minutes $MINUTES $REGIME
  echo "$SPLIT_KEY" > /out/train/split.key
fi

# ---- evaluate ---------------------------------------------------------------
cd /ws/src/kairos-suite/evaluation
echo "===== EVALUATE flow / calibration / coverage / coupling ($CONFIG) ====="
python3 -u evaluate.py --method kairos --dataset atc $SRC_ARGS --config-name $CONFIG \
  --graph "$GRAPH" --scene-dir $STUB --flow-state /out/train/flow_state.bin \
  --coverage-thresholds $COVERAGE_THRESHOLDS --minutes $MINUTES \
  --days $TEST_ARGS | tee /out/metrics_online.txt

if [ "$NO_PRESENCE" != 1 ]; then
  echo "===== EVALUATE presence ($CONFIG) ====="
  python3 -u evaluate.py --method kairos --dataset atc $SRC_ARGS --config-name $CONFIG --presence \
    --presence-horizons 5 10 60 300 600 --fremen-train-days $TRAIN_ARGS \
    --graph "$GRAPH" --scene-dir $STUB --flow-state /out/train/flow_state.bin \
    --minutes $MINUTES --days $TEST_ARGS | tee /out/presence_online.txt
fi
echo "DONE ($CONFIG) -> metrics_online.txt, presence_online.txt"
EOF

echo "[train_test] dataset=$DATASET config=$CONFIG -> $RESULTS"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  -e TBB_NUM_THREADS=1 \
  -e DATASET="$DATASET" -e CONFIG="$CONFIG" -e GRAPH="$GRAPH" \
  -e TRAIN_DAYS="$TRAIN_DAYS" -e TEST_DAYS="$TEST_DAYS" \
  -e COVERAGE_THRESHOLDS="$COVERAGE_THRESHOLDS" -e NO_PRESENCE="$NO_PRESENCE" \
  -e GODS_EYE="$GODS_EYE" -e DECIMATE="$DECIMATE" -e FOV_M="$FOV_M" -e MINUTES="$MINUTES" \
  -e HOST_UGID="$(id -u):$(id -g)" -e CHAIN="$CHAIN" \
  -v "$SUITE":/ws/src/kairos-suite \
  -v "$DATA_ROOT":/data:ro -v "$RESULTS":/out \
  "$IMAGE" bash -c 'bash -c "$CHAIN"; rc=$?; chown -R "$HOST_UGID" /out 2>/dev/null || true; exit $rc'
echo "[train_test] launched '$NAME'. Follow with:  docker logs -f $NAME"
