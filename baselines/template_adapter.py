r"""Skeleton adapter. Copy to ``<yourmethod>_adapter.py`` and fill in.

Then run:

    python3 evaluation/evaluate.py --method yourmethod --dataset atc \
        --days <held-out npz...> [your own options]

Nothing in the evaluator needs changing: it finds the adapter by name, and your
method is scored on exactly the same rows, with the same metrics, as every
other one. The full contract is in ``baselines/__init__.py``.

This file is not itself a usable method; it is skipped by ``available()``.
"""
import numpy as np

from . import Scorer


def add_arguments(ap):
    """Register options only this method needs.

    Prefix every one with the method name so adapters cannot collide: all
    adapters' options are registered on the same parser.
    """
    ap.add_argument("--yourmethod-model", default=None,
                    help="path to a trained model to load")


def make_scorer(args, ctx):
    """Load or fit the model, then return a Scorer wrapping its score_fn.

    `ctx` carries what the evaluator already resolved: `ctx.bounds` is the map
    extent the row set is clipped to, `ctx.stream_open` opens one day of the
    training stream if you fit here, `ctx.horizon_s` is the forecast horizon.
    """
    model = load_your_model(args.yourmethod_model)

    def score(t_s, xyz, theta, rho):
        """Score one chunk of detections, in time order.

        Every array is length N. Return length-N arrays: whether the model held
        an estimate at each position, and the predictive log-density at each
        observed value. Unmatched rows are charged the uniform floor by the
        evaluator, so predicting only where you are confident gains nothing.
        """
        n = len(t_s)
        matched = np.zeros(n, bool)
        log_p_heading = np.full(n, np.nan)
        mean_rho = np.full(n, np.nan)

        for i in range(n):
            cell = model.lookup(xyz[i], t_s[i])
            if cell is None:
                continue                      # leave matched[i] False
            matched[i] = True
            log_p_heading[i] = cell.log_density(theta[i])
            mean_rho[i] = cell.mean_speed

        return {
            "matched": matched,
            "log_p": {"heading": log_p_heading,
                      "speed": np.full(n, np.nan),
                      "joint": np.full(n, np.nan)},
            "mean_rho": mean_rho,
            "std_rho": np.zeros(n),
            # Optional. Supply the mass in each 45-degree sector to be scored
            # on MPP, which puts histogram and continuous models on one footing:
            # "sector_masses": np.zeros((n, 8)),
        }

    # heading_only=True keeps the unpredicted speed and joint channels out of
    # the report rather than counting them as failures.
    return Scorer(score_fn=score, heading_only=True, handle=model)


def load_your_model(path):
    raise NotImplementedError("replace with your own loader")
