# baselines — one adapter per method

The evaluator streams a dataset and accumulates metrics. This package holds one
adapter file per method, selected with `--method`.

Every method runs through the same loop, row set, and accumulators.
Unmatched detections are charged the same uniform floor for every method.

```
__init__.py            the contract: Scorer, Context, resolve(), available()
kairos_adapter.py      reference adapter, evaluates a trained Kairos flow state
stef_adapter.py        grid baseline, fits on the training days
template_adapter.py    skeleton to copy when adding a method
runners/               train + evaluate a baseline over the grounded simulation
rescore/               per-method output rescoring to circular CRPS and MPP
```

## Adding a method

Copy `template_adapter.py` to `<name>_adapter.py` and fill in two functions:

```python
def add_arguments(ap):
    """Options only your method needs. Prefix them with the method name."""
    ap.add_argument("--yourmethod-model", default=None)

def make_scorer(args, ctx):
    """Load or fit, then return a Scorer wrapping your score_fn."""
    return Scorer(score_fn=..., heading_only=True)
```

Then:

```bash
docker/run.sh python3 evaluation/evaluate.py --method yourmethod --dataset atc --days <test npz...>
```

`available()` finds the file, and your options are registered on the
evaluator's parser.

## The score_fn contract

`score_fn(t_s, xyz, theta, rho) -> dict`, called once per chunk of detections in
time order. Inputs are length-N arrays: time in unix seconds, position in world
metres `(N, 3)`, observed heading in `[0, 2pi)`, observed planar speed in m/s.

It returns length-N arrays:

| Key | Meaning |
|---|---|
| `matched` | did the model hold an estimate at this position? Unmatched rows are evaluated against the uniform floor. |
| `log_p` | `heading` / `speed` / `joint` predictive log-density at the observed value; `nan` for a channel you do not predict |
| `mean_rho`, `std_rho` | predicted speed and its spread, for speed MAE |
| `sector_masses` | optional `(N, 8)`, mass per 45-degree heading sector, used for MPP. |
| `slot_weights` | optional `(N, K)`, for weight calibration |

Set `heading_only=True` on the `Scorer` if you predict heading but not speed: the
unpredicted channels are then left out of the report.

## Getting the baseline code

The baseline implementations live in their own repositories. Check each one out
and pass the path:

```bash
docker/run.sh python3 evaluation/evaluate.py --method stef --dataset atc \
    --stefmap-path /path/to/stefmap \
    --stef-train-days <chronological train npz...> \
    --days <held-out test npz...>
```

Each adapter's module docstring names what it expects to find there.

## The other folders

`runners/` train a baseline over the grounded simulation and evaluate it online in
one pass, feeding the same accumulators as the detections stream.
`rescore/` turns a method's own saved output into circular CRPS and MPP on the
canonical row set, for methods whose output format differs.
