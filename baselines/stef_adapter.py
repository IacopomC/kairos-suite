r"""STeF-Map adapter: a grid-based directional flow baseline.

Fits on the chronological training days and returns a score_fn matching the
contract in ``baselines/__init__.py``, so the grid baseline runs through the
same streaming loop, row set, and accumulators as every other method. It is
heading-only: speed and joint densities are not predicted and come back nan.

The STeF-Map implementation itself is not vendored here. Check it out
separately and point --stefmap-path at it.

Usage:
    evaluate.py --method stef --dataset atc \
        --stefmap-path /path/to/stefmap \
        --stef-train-days <chronological train npz...> \
        --days <held-out test npz...>

    # or reuse a fit instead of refitting:
    evaluate.py --method stef --stef-model OUT/stef_fit.pkl --days ...
"""
import pathlib
import pickle
import sys

import numpy as np

from . import Scorer


def add_arguments(ap):
    """STeF-Map's own options."""
    ap.add_argument("--stef-train-days", nargs="+", default=None,
                    help="chronological training-day npz to fit on")
    ap.add_argument("--stef-model", default=None,
                    help="load a pickled fitted FlowMap instead of fitting")
    ap.add_argument("--stef-periods-s", type=float, nargs="+",
                    default=[3600.0, 43200.0, 86400.0, 604800.0],
                    help="FreMEn candidate periods; a dataset parameter")
    ap.add_argument("--stef-cell-size", type=float, default=0.4)
    ap.add_argument("--stef-window-s", type=float, default=10.0)
    ap.add_argument("--stef-order", type=int, default=1)
    ap.add_argument("--stefmap-path", default=None,
                    help="path to the STeF-Map source tree")


def make_scorer(args, ctx):
    """Fit (or load) the model and wrap its score_fn."""
    if ctx.dataset != "atc":
        raise SystemExit("[stef] this adapter is set up for --dataset atc")
    if not (args.stef_train_days or args.stef_model):
        raise SystemExit("[stef] need --stef-train-days or --stef-model")
    if not args.stefmap_path:
        raise SystemExit("[stef] --stefmap-path must point at the STeF-Map "
                         "source tree; it is not vendored in this suite")

    score_fn = stef_scorer(args.stef_train_days or [], ctx.bounds,
                           args.stef_periods_s,
                           stream_open=ctx.stream_open,
                           cell_size=args.stef_cell_size,
                           time_window_s=args.stef_window_s,
                           fremen_order=args.stef_order,
                           stefmap_path=args.stefmap_path,
                           model_pickle=args.stef_model)
    # Heading-only: the speed and joint channels are not predicted.
    return Scorer(score_fn=score_fn, heading_only=True)


def stef_scorer(train_days, bounds, periods_s, cell_size=0.4, n_bins=8,
                time_window_s=10.0, fremen_order=1, stefmap_path=None,
                stream_open=None, model_pickle=None):
    """Fit STeF-Map on the chronological training days and return a score_fn
    with the online_evaluate contract, so the grid baseline flows through the
    IDENTICAL streaming loop, row set, and accumulators as every other method.
    The per-day npz are time-sorted, so streaming them reproduces the stock
    global-sort fit bit-identically. Heading-only: speed/joint densities and
    mean_rho are NaN; the per-detection bin masses ride in `sector_masses`,
    which routes the heading-distribution scores through the edge-aligned
    histogram path."""
    sys.path.insert(0, stefmap_path)
    sys.path.insert(0, str(_HERE.parent / "grounded_sim"))
    from atc_stream import AtcStream
    from stefmap.flow_map import FlowMap
    if model_pickle is not None:
        import pickle
        model = pickle.load(open(model_pickle, "rb"))
        st = model.summary()
        print(f"[stef] loaded fitted model from {model_pickle} "
              f"({st['n_cells']} active cells)", flush=True)
        train_days = []
    model = model if model_pickle is not None else \
        FlowMap(cell_size=cell_size, n_bins=n_bins, time_window_s=time_window_s,
                fremen_order=fremen_order, candidate_periods=list(periods_s))
    tw_ns = int(time_window_s * 1e9)
    xb = (bounds[0], bounds[1]); yb = (bounds[2], bounds[3])
    window_end_ns = None; n_train = 0; tic = time.time()
    for day in train_days:
        s = (AtcStream(day, x_bounds=xb, y_bounds=yb) if stream_open is None
             else stream_open(day, xb, yb))
        t_ns = (s.t * 1e9).astype(np.int64)
        for i in range(t_ns.size):
            ti = int(t_ns[i])
            if window_end_ns is None:
                window_end_ns = ti + tw_ns
            while ti >= window_end_ns:
                model.flush_window(window_end_ns)
                window_end_ns += tw_ns
            model.add_detection(ti, float(s.x[i]), float(s.y[i]), float(s.theta[i]))
        n_train += t_ns.size
        print(f"  [stef-train] {pathlib.Path(day).name}: {t_ns.size} dets "
              f"({n_train/(time.time()-tic):.0f}/s cum)", flush=True)
    if window_end_ns is not None:
        model.flush_window(window_end_ns)
    st = model.summary()
    print(f"[stef] trained on {n_train} dets, {st['n_cells']} active cells "
          f"in {(time.time()-tic)/60:.1f} min", flush=True)

    def score(t_s, xyz, theta, rho):
        n = len(t_s)
        matched = np.zeros(n, bool)
        lph = np.full(n, np.nan)
        masses = np.zeros((n, n_bins))
        nan = np.full(n, np.nan)
        for i in range(n):
            ti = int(t_s[i] * 1e9)
            rec = model.score_row(ti, float(xyz[i][0]), float(xyz[i][1]), float(theta[i]))
            if rec["matched"]:
                matched[i] = True
                lph[i] = rec["log_p_heading"]
                masses[i] = model.cell_distribution(ti, float(xyz[i][0]), float(xyz[i][1]))
        return {"matched": matched,
                "log_p": {"heading": lph, "speed": nan, "joint": nan},
                "mean_rho": nan, "std_rho": np.zeros(n),
                "slot_weights": None, "sector_masses": masses}
    return score
