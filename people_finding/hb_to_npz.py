r"""Convert held-out station dates to the per-day npz schema the people-finding
replay consumes (t [s], x/y [mm], ang [rad], id), one file per date, the same
schema as the ATC day files. Detections are the 10 Hz interpolated stream under the
plan-map extent, the canonical row set of every other station result.

    python3 hb_to_npz.py --data-root /data/hb-station/Data-2 \
        --station HB --start 2023-04-24 --end 2023-05-21 --out npz_hb/
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "grounded_sim"))
from hb_stream import HBDayStream, hb_day_files, station_dates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--station", default="HB")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--x-bounds", type=float, nargs=2, default=(-2.0, 35.0))
    ap.add_argument("--y-bounds", type=float, nargs=2, default=(-32.0, 2.0))
    ap.add_argument("--interp-hz", type=float, default=10.0)
    ap.add_argument("--gap-max-s", type=float, default=2.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    days = [d for d in station_dates(args.data_root, args.station)
            if args.start <= d <= args.end]
    total = 0
    for day in days:
        s = HBDayStream(hb_day_files(args.data_root, args.station, day),
                           x_bounds=tuple(args.x_bounds),
                           y_bounds=tuple(args.y_bounds),
                           interp_hz=args.interp_hz, gap_max_s=args.gap_max_s)
        n = len(s.t)
        np.savez_compressed(
            out / f"{args.station.lower()}-{day}.npz",
            t=np.asarray(s.t, dtype=np.float64),
            x=(np.asarray(s.x, dtype=np.float64) * 1000.0),
            y=(np.asarray(s.y, dtype=np.float64) * 1000.0),
            ang=np.asarray(s.theta, dtype=np.float64),
            id=np.asarray(s.id, dtype=np.int64))
        total += n
        print(f"  [npz] {day}: {n:,} detections", flush=True)
    print(f"[npz] {len(days)} days, {total:,} detections -> {out}")


if __name__ == "__main__":
    main()
