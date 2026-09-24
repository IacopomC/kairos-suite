#!/usr/bin/env bash
# Periodicity measurements behind the temporal-structure supplement.
# Independent groups so they can run concurrently:
#   ./run_periodicity_followup.sh hb    - Havlickuv Brod: weekday hourly fold
#                                         (+ dedup pair), AM/PM max-power fold,
#                                         30/60-min train-cadence folds, and the
#                                         per-hour shape of the top cells.
#   ./run_periodicity_followup.sh pb    - Pardubice: same folds.
#   ./run_periodicity_followup.sh atc   - ATC: hourly daily fold with and
#                                         without per-passage dedup (isolates the
#                                         burst-inflated noise floor), then the
#                                         Eindhoven 30-min fold with dedup.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# PY: a Python with numpy, pandas and pyarrow (pyarrow reads the Eindhoven
# parquet files).
PY="${PY:-python3}"
OUT="${OUT:-$(cd "$HERE/../.." && pwd)/results/periodicity}"
MP="$PY $HERE/measure_periodicity.py"
HP="$PY $HERE/inspect_hour_profile.py"
mkdir -p "$OUT"

run() { # run <outfile> <cmd...>
  local f="$OUT/$1"; shift
  echo "[followup] $f"
  "$@" > "$f" 2>&1 || { echo "[followup] FAILED, see $f"; exit 1; }
}

case "${1:?usage: run_periodicity_followup.sh hb|pb|atc}" in
hb)
  run hb_HB_wkday_hourly.txt      $MP --source hb --station HB --fold hour --hour-bins 24 --daytype weekday --min-speed-mps 0.5
  run hb_HB_wkday_hourly_dedup.txt $MP --source hb --station HB --fold hour --hour-bins 24 --daytype weekday --min-speed-mps 0.5 --dedup
  run hb_HB_wkday_ampm.txt        $MP --source hb --station HB --fold hour --hour-bins 2  --daytype weekday --min-speed-mps 0.5
  run hb_HB_wkday_period30.txt    $MP --source hb --station HB --fold period --period-min 30 --daytype weekday --min-speed-mps 0.5
  run hb_HB_wkday_period60.txt    $MP --source hb --station HB --fold period --period-min 60 --daytype weekday --min-speed-mps 0.5
  { $HP --source hb --station HB --daytype weekday --cells 3835 3921 3750 4007 3494 3749 3579 3836 3323 3664;
    echo; echo "===== auto-picked two-way cells =====";
    $HP --source hb --station HB --daytype weekday; } > "$OUT/hb_HB_hour_profile.txt" 2>&1
  ;;
pb)
  run hb_PB_wkday_hourly.txt      $MP --source hb --station PB --fold hour --hour-bins 24 --daytype weekday --min-speed-mps 0.5
  run hb_PB_wkday_ampm.txt        $MP --source hb --station PB --fold hour --hour-bins 2  --daytype weekday --min-speed-mps 0.5
  run hb_PB_wkday_period30.txt    $MP --source hb --station PB --fold period --period-min 30 --daytype weekday --min-speed-mps 0.5
  run hb_PB_wkday_period60.txt    $MP --source hb --station PB --fold period --period-min 60 --daytype weekday --min-speed-mps 0.5
  $HP --source hb --station PB --daytype weekday > "$OUT/hb_PB_hour_profile.txt" 2>&1
  ;;
atc)
  run atc_daily_mv.txt        $MP --source atc --fold hour --hour-bins 24 --min-speed-mps 0.5
  run atc_daily_mv_dedup.txt  $MP --source atc --fold hour --hour-bins 24 --min-speed-mps 0.5 --dedup
  run eindhoven_period30_dedup.txt $MP --source eindhoven --fold period --period-min 30 --min-speed-mps 0.5 --dedup
  ;;
misc)
  # Honest-floor (dedup) re-runs of the remaining per-frame datasets, so every
  # dataset's direction read uses the same effective-sample-size correction.
  run tbd_period1_dedup.txt   $MP --source tbd --fold period --period-min 1  --min-speed-mps 0.5 --dedup
  run tbd_period5_dedup.txt   $MP --source tbd --fold period --period-min 5  --min-speed-mps 0.5 --dedup
  run tbd_period10_dedup.txt  $MP --source tbd --fold period --period-min 10 --min-speed-mps 0.5 --dedup
  run edinburgh_wkday_hourly_dedup.txt $MP --source edinburgh --fold hour --hour-bins 24 --daytype weekday --min-speed-mps 0.5 --dedup
  run atc_weekly_dedup.txt    $MP --source atc --fold dow --min-speed-mps 0.5 --dedup
  run atc_hour_profile.txt    $HP --source atc --daytype all --min-speed-mps 0.5
  ;;
speed)
  # Speed-channel measurement (the tool reports SPEED MIX in every run): how
  # much per-slot speed varies with time, the evidence for holding the per-slot
  # speed mean time-invariant.
  run atc_daily_speedch.txt   $MP --source atc --fold hour --hour-bins 24 --min-speed-mps 0.5 --dedup
  run hb_HB_wkday_speedch.txt $MP --source hb --station HB --fold hour --hour-bins 24 --daytype weekday --min-speed-mps 0.5 --dedup
  run tbd_period5_speedch.txt $MP --source tbd --fold period --period-min 5 --min-speed-mps 0.5 --dedup
  ;;
*) echo "unknown group: $1" >&2; exit 1 ;;
esac
echo ALLDONE
