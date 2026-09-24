"""Method adapters for the evaluator.

The evaluator (``evaluation/evaluate.py``) knows how to stream a dataset's
held-out detections and accumulate metrics over them. It knows nothing about
any particular flow model. The seam between the two is one function per
method, living in this package as ``<name>_adapter.py``, so scoring a new method
means adding one file and passing ``--method <name>``. Nothing in the
evaluator changes.

Every method therefore runs through the identical streaming loop, the identical
row set, and the identical accumulators. A comparison reflects the models, not
the harness.

An adapter module defines two things:

    def add_arguments(parser):
        '''Register this method's own options. Prefix them with the method
        name so they cannot collide: --mymethod-model, --mymethod-path.'''

    def make_scorer(args, ctx) -> Scorer:
        '''Load or fit the model and return a Scorer wrapping its score_fn.'''

``ctx`` carries what the evaluator already worked out: see :class:`Context`.

The score_fn contract
---------------------
``score_fn(t_s, xyz, theta, rho) -> dict`` is called once per chunk of
detections, in time order. Every array is length N, one entry per detection:

    t_s      float64  detection time, unix seconds
    xyz      (N, 3)   detection position in world metres
    theta    float64  observed heading in [0, 2pi)
    rho      float64  observed planar speed, m/s

It returns a dict of length-N arrays:

    matched   bool     did the model hold an estimate at this position?
                       Unmatched rows are scored against the uniform floor, so
                       a model gains nothing by predicting only where it is
                       confident.
    log_p     dict     'heading', 'speed', 'joint' -> predictive log-density at
                       the observed value. Use nan for a channel the model does
                       not predict and set Scorer.heading_only.
    mean_rho  float64  predicted speed, for speed MAE
    std_rho   float64  its standard deviation, or zeros

Optional keys, used by metrics that need them:

    sector_masses  (N, 8)  mass in each 45-degree heading sector, for MPP and
                           for scoring histogram models on the same footing as
                           continuous ones
    slot_weights   (N, K)  per-slot mixing weights, for weight calibration

Adding a method
---------------
Copy ``template_adapter.py`` to ``<name>_adapter.py``, fill in the two functions, and
run with ``--method <name>``. ``available()`` picks it up automatically.
"""
from __future__ import annotations

import importlib
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_HERE = pathlib.Path(__file__).resolve().parent


@dataclass
class Context:
    """What the evaluator has already resolved when it builds a scorer.

    bounds       (xmin, xmax, ymin, ymax) map extent the row set is clipped to,
                 or None for an unclipped dataset.
    dataset      dataset key, e.g. 'atc' or 'tbd'.
    stream_open  callable opening one day of the training stream, for methods
                 that fit at construction time. None when the method loads a
                 state instead of fitting.
    horizon_s    prediction horizon the forecast is queried at.
    """

    bounds: Optional[tuple] = None
    dataset: str = ""
    stream_open: Optional[Callable] = None
    horizon_s: float = 0.0


@dataclass
class Scorer:
    """One method, ready to score.

    score_fn      the callable described in the module docstring.
    heading_only  True if the method predicts heading but not speed, so the
                  speed and joint channels are left out of the report rather
                  than counted as failures.
    coupling_json unused by most methods; carries a model's own edge-coupling
                  state when it has one.
    handle        anything that must stay alive for the run (an open pipeline,
                  a loaded model). Held only so it is not garbage-collected.
    """

    score_fn: Callable
    heading_only: bool = False
    coupling_json: Optional[str] = None
    handle: Any = field(default=None, repr=False)


def available() -> list:
    """Method names with an adapter in this package, e.g. ['kairos', 'stef']."""
    return sorted(p.stem[:-len("_adapter")] for p in _HERE.glob("*_adapter.py")
                  if not p.stem.startswith(("_", "template")))


def resolve(name: str):
    """Import the adapter module for `name`, or fail with the known names."""
    if name not in available():
        raise SystemExit(
            f"[baselines] no adapter for '{name}'. Available: "
            f"{', '.join(available())}. Add one by copying "
            f"baselines/template_adapter.py to baselines/{name}_adapter.py.")
    return importlib.import_module(f"{__name__}.{name}_adapter")
