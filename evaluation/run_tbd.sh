#!/usr/bin/env bash
# TBD, end to end, in the kairos-suite image: train the flow model through the
# training sessions in chronological order (each session resumes the previous
# session's flow state; the 3DSG is rebuilt per session), then score every test
# session against the final state and pool the test predictions into one
# metric report.
#
#   DATA_ROOT=/path/to/datasets ./run_tbd.sh
#   docker logs -f kairos-tbd                 # follow the run
#
# Data layout under $DATA_ROOT (mounted read-only at /data): one FileDataLoader
# scene per session under tbd/<YYYY-MM>/, each holding its pedestrian tracks:
#   tbd/2022-12/scene_tbd_2022-12-07-12-15-36/{...scene files..., tracks.csv}
# The default split is the paper's: 10 training sessions (December 2022 and
# January 2023) and 6 test sessions (February 2023), listed below. Override
# with TRAIN_SCENES / TEST_SCENES (paths relative to $DATA_ROOT/tbd).
#
# Output under $RESULTS: train/<session>/temporal/flow_state.bin per session,
# test/<session>/preds.csv per test session, preds_pooled.csv and
# metrics_online.txt (the flow-prediction table's columns).
#
# Env: DATA_ROOT, RESULTS, IMAGE, TRAIN_SCENES, TEST_SCENES, TRACKS_FILE
#   (name of the tracks CSV inside each scene, default tracks.csv), NAME.
set -euo pipefail

SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"
RESULTS="${RESULTS:-$SUITE/results/tbd}"
IMAGE="${IMAGE:-kairos-suite}"
TRACKS_FILE="${TRACKS_FILE:-tracks.csv}"
NAME="${NAME:-kairos-tbd}"
TRAIN_SCENES="${TRAIN_SCENES:-2022-12/scene_tbd_2022-12-07-12-15-36 2022-12/scene_tbd_2022-12-07-12-44-46
2022-12/scene_tbd_2022-12-07-13-21-55 2022-12/scene_tbd_2022-12-07-13-46-01
2023-01/scene_tbd_2023-01-23-13-15-54 2023-01/scene_tbd_2023-01-23-13-52-34
2023-01/scene_tbd_2023-01-23-14-16-02 2023-01/scene_tbd_2023-01-26-12-54-28
2023-01/scene_tbd_2023-01-26-13-36-59 2023-01/scene_tbd_2023-01-26-15-17-19}"
TEST_SCENES="${TEST_SCENES:-2023-02/scene_tbd_2023-02-01-12-48-21 2023-02/scene_tbd_2023-02-01-13-02-13
2023-02/scene_tbd_2023-02-01-13-51-05 2023-02/scene_tbd_2023-02-02-12-05-25
2023-02/scene_tbd_2023-02-02-12-14-22 2023-02/scene_tbd_2023-02-03-11-42-50}"

[[ -d "$DATA_ROOT/tbd" ]] || { echo "$DATA_ROOT/tbd does not exist (set DATA_ROOT)"; exit 1; }
for s in $TRAIN_SCENES $TEST_SCENES; do
  [[ -f "$DATA_ROOT/tbd/$s/$TRACKS_FILE" ]] || { echo "missing $DATA_ROOT/tbd/$s/$TRACKS_FILE"; exit 1; }
done
mkdir -p "$RESULTS"

read -r -d '' CHAIN <<'EOF' || true
set -euo pipefail
RUN="kairos run -c tbd -l ade20k_outdoor --no-publish --no-enable-lcd"

echo "===== TRAIN (chained over $(echo $TRAIN_SCENES | wc -w) sessions) ====="
prev=""
for s in $TRAIN_SCENES; do
  scene=/data/tbd/$s; name=$(basename "$s")
  $RUN "$scene" -o /out/train --tracks "$scene/$TRACKS_FILE" ${prev:+--load-flow-state "$prev"}
  prev=/out/train/$name/temporal/flow_state.bin
  [ -f "$prev" ] || { echo "training on $name wrote no $prev"; exit 1; }
done
echo "[train] final state: $prev"

echo "===== SCORE test sessions ====="
for s in $TEST_SCENES; do
  scene=/data/tbd/$s; name=$(basename "$s")
  $RUN "$scene" -o /out/test --load-flow-state "$prev" --score-tracks "$scene/$TRACKS_FILE"
  [ -f "/out/test/$name/preds.csv" ] || { echo "scoring $name wrote no preds.csv"; exit 1; }
done

# Pool the test sessions (one shared world frame) into a single file, header once.
first=1
for s in $TEST_SCENES; do
  f=/out/test/$(basename "$s")/preds.csv
  if [ $first = 1 ]; then cat "$f"; first=0; else tail -n +2 "$f"; fi
done > /out/preds_pooled.csv
COUPLING=/out/test/$(basename $(echo $TEST_SCENES | awk '{print $1}'))/preds.coupling.json
CPL=""; [ -f "$COUPLING" ] && CPL="--coupling-json $COUPLING"

python3 -u /ws/src/kairos-suite/evaluation/score_preds.py --preds /out/preds_pooled.csv \
  --coverage-thresholds 10 50 $CPL --out /out/metrics_online.txt
echo "DONE (tbd) -> metrics_online.txt"
EOF

echo "[run_tbd] -> $RESULTS"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  -e TBB_NUM_THREADS=1 \
  -e TRAIN_SCENES="$TRAIN_SCENES" -e TEST_SCENES="$TEST_SCENES" -e TRACKS_FILE="$TRACKS_FILE" \
  -e HOST_UGID="$(id -u):$(id -g)" -e CHAIN="$CHAIN" \
  -v "$SUITE":/ws/src/kairos-suite \
  -v "$DATA_ROOT":/data:ro -v "$RESULTS":/out \
  "$IMAGE" bash -c 'bash -c "$CHAIN"; rc=$?; chown -R "$HOST_UGID" /out 2>/dev/null || true; exit $rc'
echo "[run_tbd] launched '$NAME'. Follow with:  docker logs -f $NAME"
