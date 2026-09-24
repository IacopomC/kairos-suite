#!/usr/bin/env bash
# Build a scrubbable Rerun recording (.rrd) of a Kairos grounded simulation, for
# any dataset. Two decoupled stages (different Python versions -> two images):
#   Stage 1 (kairos image, Python 3.8):  run_viz.py     -> flow.npz
#   Stage 2 (kairos-viz image, Python 3.11): render     -> out.rrd
# The only dataset-specific inputs are the graph, the stream source, and the days.
#
#   # ATC, model-building drive over one day:
#   STREAM_SOURCE=atc GRAPH=/data/atc/graph/atc_graph.json \
#     DAYS="/data/atc/npz/atc-20121114.npz" ./create_rrd.sh
#
#   # HB, prediction over held-out days from a frozen state:
#   STREAM_SOURCE=hb STATION=HB MODE=predict \
#     GRAPH=/data/hb-station/graph/hb_flow_graph.json \
#     FLOW_STATE=/state/flow_state.bin STATE_DIR=results/hb/train \
#     START=2023-04-24 END=2023-04-30 ./create_rrd.sh
set -euo pipefail

STREAM_SOURCE="${STREAM_SOURCE:-atc}"        # atc | hb
MODE="${MODE:-online}"                       # online (build model) | predict (frozen)
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"
IMAGE="${IMAGE:-kairos-suite}"
KAIROS_DIR="${KAIROS_DIR:-$SUITE/../kairos}"    # kairos checkout (holds kairos_visualizer)
VIZ_IMAGE="${VIZ_IMAGE:-kairos-viz}"
GRAPH="${GRAPH:?set GRAPH to the authored graph .json (in-container /data path)}"
OCCUPANCY="${OCCUPANCY:-}"                    # authored occupancy-layer .npz (walls)
PREBUILT="${PREBUILT:-}"                       # online: static map + warm-started model
SCENE="${SCENE:-stub_scene}"
CONFIG="${CONFIG:-atc}"
STATION="${STATION:-HB}"                      # hb only
HB_ROOT="${HB_ROOT:-/data/hb-station/Data-2}"
FLOW_STATE="${FLOW_STATE:-}"                  # required for MODE=predict (in-container path)
STATE_DIR="${STATE_DIR:-}"                    # host folder mounted at /state (holds FLOW_STATE)
DAYS="${DAYS:-}"                              # atc: npz/csv paths; hb: ISO dates
START="${START:-}"; END="${END:-}"            # hb: resolve all-camera dates in range
OUT="${OUT:-$SUITE/results/viz}"
RENDER_ARGS="${RENDER_ARGS:-}"                # extra stage-2 flags, e.g. --arrow-color heading
# cadence knobs (see run_viz.py --help)
SNAP="${SNAP:-1200}"; FINE_SNAP="${FINE_SNAP:-0}"; FINE_UNTIL_H="${FINE_UNTIL_H:-0}"
QUERY_EVERY="${QUERY_EVERY:-120}"; DET_DECIMATE="${DET_DECIMATE:-1}"; MINUTES="${MINUTES:-0}"
UID_GID="$(id -u):$(id -g)"
mkdir -p "$OUT"
STATE_MOUNT=""
if [ -n "$STATE_DIR" ]; then
  case "$STATE_DIR" in /*) ;; *) STATE_DIR="$SUITE/$STATE_DIR";; esac
  STATE_MOUNT="-v $STATE_DIR:/state:ro"
fi

if [ "$MODE" = predict ] && [ -z "$FLOW_STATE" ]; then
  echo "create_rrd: MODE=predict needs FLOW_STATE=<frozen state path>"; exit 1
fi

# Always rebuild (Docker layer cache makes this a no-op when the visualizer
# source is unchanged) so edits to the renderer are never silently skipped.
echo "[create_rrd] building $VIZ_IMAGE from kairos/kairos_visualizer (cached if unchanged)"
docker build -t "$VIZ_IMAGE" "$KAIROS_DIR/kairos_visualizer"

echo "[create_rrd] stage 1 ($STREAM_SOURCE / $MODE) -> flow.npz"
docker run --rm -v "$SUITE":/ws/src/kairos-suite $STATE_MOUNT \
  -v "$DATA_ROOT":/data:ro -v "$OUT":/out \
  -e HOST_UGID="$UID_GID" \
  -e STREAM_SOURCE="$STREAM_SOURCE" -e MODE="$MODE" -e GRAPH="$GRAPH" -e SCENE="$SCENE" \
  -e CONFIG="$CONFIG" -e STATION="$STATION" -e HB_ROOT="$HB_ROOT" \
  -e FLOW_STATE="$FLOW_STATE" -e OCCUPANCY="$OCCUPANCY" -e PREBUILT="$PREBUILT" \
  -e DAYS="$DAYS" -e START="$START" -e END="$END" \
  -e SNAP="$SNAP" -e FINE_SNAP="$FINE_SNAP" -e FINE_UNTIL_H="$FINE_UNTIL_H" \
  -e QUERY_EVERY="$QUERY_EVERY" -e DET_DECIMATE="$DET_DECIMATE" -e MINUTES="$MINUTES" \
  "$IMAGE" bash -c '
    set -e; set +u; source /opt/ros/noetic/setup.bash; source /ws/devel/setup.bash; set -u
    export PYTHONPATH=/ws/src/kairos/python/src:$PYTHONPATH
    cd /ws/src/kairos-suite/grounded_sim
    DD="$DAYS"
    if [ -z "$DD" ] && [ "$STREAM_SOURCE" = hb ]; then
      DD=$(python3 -c "from hb_stream import station_dates; print(\" \".join(d for d in sorted(station_dates(\"$HB_ROOT\",\"$STATION\")) if \"$START\"<=d<=\"$END\"))")
    fi
    [ -n "$DD" ] || { echo "no days: set DAYS=... (atc paths) or START/END (hb)"; exit 1; }
    FS=""; [ -n "$FLOW_STATE" ] && FS="--flow-state $FLOW_STATE"
    OCC=""; [ -n "$OCCUPANCY" ] && OCC="--occupancy $OCCUPANCY"
    PB=""; [ -n "$PREBUILT" ] && PB="--prebuilt-map"
    python3 -u run_viz.py --graph "$GRAPH" --days $DD --scene-dir "$SCENE" $OCC $PB \
      --config-name "$CONFIG" --stream-source "$STREAM_SOURCE" --station "$STATION" \
      --hb-root "$HB_ROOT" --mode "$MODE" $FS \
      --snapshot-every-s "$SNAP" --fine-snapshot-s "$FINE_SNAP" --fine-until-h "$FINE_UNTIL_H" \
      --query-every-s "$QUERY_EVERY" --det-decimate "$DET_DECIMATE" --minutes "$MINUTES" \
      --out-npz /out/flow.npz
    chown -R $HOST_UGID /out 2>/dev/null || true'

echo "[create_rrd] stage 2: render out.rrd"
docker run --rm --user "$UID_GID" -v "$OUT":/out "$VIZ_IMAGE" /out/flow.npz -o /out/out.rrd $RENDER_ARGS
echo "[create_rrd] wrote $OUT/out.rrd  (open with: rerun $OUT/out.rrd)"
