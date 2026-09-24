#!/usr/bin/env bash
# Run one command inside the kairos-suite image, from this repository's
# checkout, with your datasets at /data (read-only) and a results folder at
# /out. Paths inside the command are in-container paths.
#
#   DATA_ROOT=/path/to/datasets docker/run.sh python3 evaluation/evaluate.py --help
#   docker/run.sh bash                              # interactive shell
#
# Env: DATA_ROOT (default ~/datasets), RESULTS_ROOT (default <checkout>/results),
#   IMAGE (default kairos-suite).
set -euo pipefail

SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-$SUITE/results}"
IMAGE="${IMAGE:-kairos-suite}"
mkdir -p "$RESULTS_ROOT"

TTY=""; [ -t 0 ] && [ -t 1 ] && TTY="-it"
exec docker run --rm $TTY -e TBB_NUM_THREADS=1 \
  -v "$SUITE":/ws/src/kairos-suite \
  -v "$DATA_ROOT":/data:ro -v "$RESULTS_ROOT":/out \
  -w /ws/src/kairos-suite \
  "$IMAGE" "$@"
