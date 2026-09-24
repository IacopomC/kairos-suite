# dataset-analysis — what can any method achieve on this data?

These measure the data itself. They answer questions like *is there a daily
rhythm in walking direction here at all*, *how much of the heading error is
tracking noise and how much is forecast error*.

Methods are compared against each other in [`../baselines/`](../baselines/).

If you are bringing in a new dataset, start with `periodicity/`.

## periodicity/ — is there recurrent structure, and at what period?

**`measure_periodicity.py`** is the main tool. It asks whether the direction mix
of a cell departs from its all-day average by hour, and it carries its own null:
even-day and odd-day histograms (*same hour, different days*) give a split-half
sampling-noise floor.
The reported rhythm index is across-hour signal over within-hour noise, and a
rhythm is real only when that ratio is well above 1.

```bash
# ATC, weekday direction rhythm, one vote per crossing
python3 measure_periodicity.py --source atc --fold hour --hour-bins 24 \
    --daytype weekday --min-speed-mps 0.5 --dedup

# a short recurrence, here a 30-minute period
python3 measure_periodicity.py --source eindhoven --fold period --period-min 30
```

Use `--dedup` for the direction channel. At 10 fps a single crossing otherwise
casts tens of identical votes and a loiterer thousands, which inflates the noise
floor well above true sampling noise. Occupancy stays per-frame either way.

Voxel size, direction bins, hour bins and the busiest-cell count are all flags.

**`inspect_hour_profile.py`** runs after the above, once you know a rhythm
exists, to see *which* mechanism produces it. A reversal (the dominant
direction's share swinging across the day, the commuter pattern) and a
dispersion (the mix simply broadening) give the same divergence signature but
mean opposite things for forecasting: only the first is predictable from the
hour.

**`inspect_cycle.py`** checks whether a short cycle is a real cadence, for
instance a train timetable, by folding into phase bins and looking for a sharp
surge. Configured by module constants at the top of the file.

**`run_periodicity_followup.sh`** drives the above across datasets and folds.

**`edinburgh_stream.py`, `eindhoven_stream.py`** are loaders for the Edinburgh
and Eindhoven datasets.

**Adding a dataset:** write a loader and add a branch in the two places
`--source` is tested in `measure_periodicity.py` (`open_stream`, and the
day-listing branch in `main`).

## noise_ceiling.py — how much of the metric is measurement noise?

Evaluates the same saved predictions twice, against raw headings and against
headings recomputed from time-smoothed positions. The gap is tracking jitter,
so it bounds how much of the remaining heading metric
any model could recover.

```bash
python3 noise_ceiling.py --preds results/<run>/preds.csv
```

The smoothing constants are set for 10 Hz tracking; raise `SMOOTH_S` and
`DIFF_S` for a slower stream.

## Running these

The scripts run in the `kairos-suite` image. Paths come from the environment:

```bash
DATA_ROOT=/path/to/datasets docker/run.sh bash   # from the suite checkout
export DATA_ROOT=/data                           # dataset root
export KAIROS_RESULTS=/out                       # where runs wrote their output
```

Several scripts are set up for ATC or HB, with their per-dataset details in
module constants at the top of the file.
Each one's docstring says which constants to change.
