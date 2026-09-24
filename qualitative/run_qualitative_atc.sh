#!/usr/bin/env bash
# Qualitative panels of a trained ATC flow state, end to end: read the state
# out per hour of one day (export_flow_mod.py, in the kairos image), then draw
# the panels (plot_mod_panels.py, outside it).
#
#   STATE=results/atc/train/flow_state.bin \
#   DATE=2013-09-01 REGIONS="open=-20,0,-10,10 corridor=15,25,-17,-7" \
#   ./run_qualitative_atc.sh
#
# PITCHES may name more than one bin size, each read out and drawn into its own
# subdirectory: the mean field reads best at the cell size the other maps use,
# while a mixture spread over many directions needs a coarser bin to be legible.
#
# Env: CONFIG (the config the state was trained under), HOURS, MINUTE, PITCHES,
#   UTC_OFFSET_H, MIN_WEIGHT, WEIGHT_LENGTH (1 to shorten a component by its
#   weight), LABELS (1 to title each panel), AXES (1 to keep the metre axes),
#   OUT, GRAPH and MAP_YAML (container paths), DATA_ROOT, IMAGE,
#   RENDER_PY.
set -euo pipefail

SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"
IMAGE="${IMAGE:-kairos-suite}"
RENDER_PY="${RENDER_PY:-python3}"
CONFIG="${CONFIG:-atc}"
GRAPH="${GRAPH:-/data/atc/graph/atc_graph.json}"
MAP_YAML="${MAP_YAML:-/data/atc/ATC-map/localization_grid.yaml}"
HOURS="${HOURS:-9-20}"
MINUTE="${MINUTE:-30}"
PITCHES="${PITCHES:-1.0}"
MIN_WEIGHT="${MIN_WEIGHT:-0.05}"
WEIGHT_LENGTH="${WEIGHT_LENGTH:-0}"
UTC_OFFSET_H="${UTC_OFFSET_H:-9}"
LABELS="${LABELS:-0}"
AXES="${AXES:-0}"
: "${STATE:?set STATE (path to flow_state.bin, absolute or relative to the repo)}"
: "${DATE:?set DATE (the day to read out, ISO, e.g. 2013-09-01)}"
: "${REGIONS:?set REGIONS (space-separated name=x0,x1,y0,y1)}"

case "$STATE" in /*) STATE_ABS="$STATE";; *) STATE_ABS="$SUITE/$STATE";; esac
[[ -f "$STATE_ABS" ]] || { echo "no state at $STATE_ABS" >&2; exit 1; }
OUT="${OUT:-$(dirname "$STATE_ABS")/qualitative_$DATE}"
mkdir -p "$OUT"

REGION_FLAGS=""; for r in $REGIONS; do REGION_FLAGS="$REGION_FLAGS --region $r"; done

NAME="${NAME:-kairos_qualitative}"
echo "[qualitative] state=$STATE_ABS  date=$DATE  regions=[$REGIONS]  -> $OUT"
# The plan the panels are drawn on lives with the dataset, not in the image.
HOST_MAP="${MAP_YAML/\/data/$DATA_ROOT}"
WL=""; [[ "$WEIGHT_LENGTH" = "1" ]] && WL="--weight-length"
AX=""; [[ "$AXES" = "1" ]] && AX="--axes"

for pitch in $PITCHES; do
  sub="$OUT/pitch_$pitch"
  mkdir -p "$sub"
  read -r -d '' CHAIN <<EOF || true
set -e
set +u; source /opt/ros/noetic/setup.bash; source /ws/devel/setup.bash; set -u
export PYTHONPATH=/ws/src/kairos/python/src:\$PYTHONPATH
cd /ws/src/kairos-suite/qualitative
python3 -u export_flow_mod.py --flow-state /state/$(basename "$STATE_ABS") \\
  --graph $GRAPH --scene-dir /ws/src/kairos-suite/grounded_sim/stub_scene \\
  --config-name $CONFIG --date $DATE --hours $HOURS --minute $MINUTE \\
  --utc-offset-h $UTC_OFFSET_H --pitch $pitch $REGION_FLAGS --out-dir /out/mods
chown -R $(id -u):$(id -g) /out 2>/dev/null || true
EOF
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run --rm --name "$NAME" -e TBB_NUM_THREADS=1 \
    -v "$SUITE":/ws/src/kairos-suite \
    -v "$DATA_ROOT":/data:ro -v "$(dirname "$STATE_ABS")":/state:ro -v "$sub":/out \
    "$IMAGE" bash -c "$CHAIN"

  for r in $REGIONS; do
    name="${r%%=*}"; box="${r#*=}"
    label=""; [[ "$LABELS" = "1" ]] && label="$name"
    "$RENDER_PY" -u "$SUITE/qualitative/plot_mod_panels.py" \
      --mods "$sub/mods/$name" --map-yaml "$HOST_MAP" --crop="$box" \
      --minute "$MINUTE" --label="$label" --min-weight "$MIN_WEIGHT" $WL $AX \
      --out-dir "$sub/panels_$name"
  done
done
echo "[qualitative] done -> $OUT/pitch_*/panels_*"
