#!/usr/bin/env bash
# Fast Online CLiFF-map flow baseline on ATC (heading circular-CRPS + MPP).
# Two stages, same train/test detections as the rheos/aion ATC baselines:
#   A) Docker (kairos-suite): dump the robot field-of-view TRAINING stream
#      (84 authored train days) to online CLiFF-map CSVs + bounds.json.
#   B) venv: online build + score the 8 held-out test days.
# WARNING: the full run is multi-hour (Stage A streams 84 full days at 10 Hz;
# Stage B scores all held-out detections). Set MINUTES for a quick slice.
# Published online CLiFF-map grid resolution is 1 m; override with RADIUS/STEP=0.4
# for a resolution-matched comparison against the 0.4 m batch baselines. The
# Stage A stream is resolution-independent, so SKIP_EXTRACT=1 reuses an existing
# EXTRACT_DIR and goes straight to the build.
set -euo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA_ROOT:-$HOME/datasets}"
IMAGE="${IMAGE:-kairos-suite}"
VENV="${VENV:?set VENV to the python interpreter of the baseline method venv}"
EXTRACT_DIR="${EXTRACT_DIR:-$SUITE/results/online_cliff/atc_extract}"
OUT="${OUT:-$SUITE/results/online_cliff/heading_dist_atc.txt}"
MINUTES="${MINUTES:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
REGIME="${REGIME:-embodied}"   # embodied | fullvis
# Canonical ATC occupancy-map extent: the rectangle every other ATC row scores,
# wider than the embodied graph box recorded in the extract's bounds.json.
ATC_X_BOUNDS="${ATC_X_BOUNDS:--60 80}"
ATC_Y_BOUNDS="${ATC_Y_BOUNDS:--40 20}"
GRAPH=/data/atc/graph/atc_graph.json
EXTRA=""
[[ -n "${RADIUS:-}" ]] && EXTRA="$EXTRA --radius $RADIUS"
[[ -n "${STEP:-}" ]] && EXTRA="$EXTRA --step $STEP"
# Persist the fitted map so a metric change is a re-score, not a retrain.
[[ -n "${WORK_DIR:-}" ]] && EXTRA="$EXTRA --work-dir $WORK_DIR"

# Train days = the 84 authored days (provenance); test = the remaining 8 npz.
mapfile -t TEST < <(PROV="$DATA/atc/graph/atc_graph.provenance.json" NPZ_GLOB="$DATA/atc/npz/*.npz" "$VENV" - <<'PY'
import json,glob,os
train=set(json.load(open(os.environ["PROV"]))["days"])
alln=sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.environ["NPZ_GLOB"]))
print("\n".join(d for d in alln if d not in train))
PY
)
TEST_NPZ=""; for d in "${TEST[@]}"; do TEST_NPZ="$TEST_NPZ $DATA/atc/npz/$d.npz"; done
mkdir -p "$EXTRACT_DIR" "$(dirname "$OUT")"

if [[ "$SKIP_EXTRACT" == 1 ]]; then
  echo "[atc] Stage A: skipped, reusing $EXTRACT_DIR"
else
  mapfile -t TRAIN < <(PROV="$DATA/atc/graph/atc_graph.provenance.json" NPZ_GLOB="$DATA/atc/npz/*.npz" "$VENV" - <<'PY'
import json,os
print("\n".join(json.load(open(os.environ["PROV"]))["days"]))
PY
)
  TRAIN_NPZ=""; for d in "${TRAIN[@]}"; do TRAIN_NPZ="$TRAIN_NPZ /data/atc/npz/$d.npz"; done
  echo "[atc] Stage A: extract FoV training stream (${#TRAIN[@]} days) -> $EXTRACT_DIR"
  docker run --rm --name online_cliff_atc_extract \
    -v "$SUITE":/ws/src/kairos-suite \
    -v "$DATA":/data:ro -v "$EXTRACT_DIR":/out \
    "$IMAGE" bash -c "set +u; source /opt/ros/noetic/setup.bash; source /ws/devel/setup.bash; export PYTHONPATH=/ws/src/kairos/python/src:\$PYTHONPATH; set -u; \
      python3 -u /ws/src/kairos-suite/baselines/rescore/extract_fov_stream_online_cliff.py \
        --graph $GRAPH --days$TRAIN_NPZ --out-dir /out --minutes $MINUTES; \
      chown -R $(id -u):$(id -g) /out"
fi

if [[ "$REGIME" == fullvis ]]; then
  mapfile -t TRAIN_FV < <(PROV="$DATA/atc/graph/atc_graph.provenance.json" NPZ_GLOB="$DATA/atc/npz/*.npz" "$VENV" - <<'PYFV'
import json,os
print("\n".join(json.load(open(os.environ["PROV"]))["days"]))
PYFV
)
  TRAIN_NPZ_FV=""; for d in "${TRAIN_FV[@]}"; do TRAIN_NPZ_FV="$TRAIN_NPZ_FV $DATA/atc/npz/$d.npz"; done
  echo "[atc] Stage B (full visibility): ${#TRAIN_FV[@]} train days -> $OUT"
  "$VENV" -u "$SUITE/baselines/rescore/rescore_heading_dist_online_cliff_fullvis.py" \
    --source atc --bounds "$EXTRACT_DIR/bounds.json" \
    --x-bounds $ATC_X_BOUNDS --y-bounds $ATC_Y_BOUNDS \
    --train-days $TRAIN_NPZ_FV --test-days $TEST_NPZ --out "$OUT" $EXTRA
else
  echo "[atc] Stage B (embodied): online build + score ${#TEST[@]} test days -> $OUT"
  "$VENV" -u "$SUITE/baselines/rescore/rescore_heading_dist_online_cliff_embodied.py" \
    --extract-dir "$EXTRACT_DIR" --test-days $TEST_NPZ --out "$OUT" $EXTRA
fi
echo "[atc] done -> $OUT"
