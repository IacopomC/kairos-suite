#!/usr/bin/env python3
"""Look at the shape of the ~30-min Eindhoven cycle: is it a real train cadence,
and which way do people move at each phase? Loads a couple of 10-day files (enough
to see the shape), folds movers into 30 one-minute phase bins, and prints:
  * global occupancy per phase (a train arrival should be a sharp surge);
  * for the busiest TWO-WAY cells, the along-platform split (+x vs -x) per phase,
    so a directional swing shows up as the split changing across the cycle.

Usage:
    DATA_ROOT=~/datasets python3 inspect_cycle.py

Reads the first two day files it finds. Configured by the module constants at
the top rather than by flags: PERIOD_S is the cycle length under test, NPH the
phase bins, XB/YB the region, MINSPD the mover threshold. Change those for a
different site or a different candidate cycle.
"""
import glob, math, sys, os
import os
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eindhoven_stream import EindhovenStream

PERIOD_S = 1800.0
NPH = 30                                   # 1-min bins
XB = (-10.0, 80.0); YB = (-12.0, 10.0); VS = 0.4
MINSPD = 0.5

files = sorted(glob.glob(os.path.join(DATA_ROOT, "eindhoven-station", "*.parquet")))[:2]
nx = int(math.ceil((XB[1]-XB[0])/VS)); ny = int(math.ceil((YB[1]-YB[0])/VS))
occ_ph = np.zeros(NPH)                      # global occupancy per phase (all det)
# per-cell along-platform components per phase (moving only)
xpos = np.zeros((nx*ny, NPH)); xneg = np.zeros((nx*ny, NPH))
ypos = np.zeros((nx*ny, NPH)); yneg = np.zeros((nx*ny, NPH))
cellmov = np.zeros(nx*ny)

for f in files:
    s = EindhovenStream(f, x_bounds=XB, y_bounds=YB)
    ix = np.clip(((s.x-XB[0])/VS).astype(np.int64),0,nx-1)
    iy = np.clip(((s.y-YB[0])/VS).astype(np.int64),0,ny-1)
    cell = ix*ny+iy
    iph = np.clip((np.mod(s.t,PERIOD_S)/(PERIOD_S/NPH)).astype(np.int64),0,NPH-1)
    occ_ph += np.bincount(iph, minlength=NPH)
    mv = np.isfinite(s.theta) & (s.rho>MINSPD)
    c=cell[mv]; p=iph[mv]; th=s.theta[mv]
    # along-platform (x): cos(theta) sign; across (y): sin(theta) sign
    fx=np.cos(th); fy=np.sin(th)
    np.add.at(xpos, (c[fx>0], p[fx>0]), 1); np.add.at(xneg, (c[fx<0], p[fx<0]), 1)
    np.add.at(ypos, (c[fy>0], p[fy>0]), 1); np.add.at(yneg, (c[fy<0], p[fy<0]), 1)
    np.add.at(cellmov, c, np.ones(c.size))
    print(f"  loaded {os.path.basename(f)}", flush=True)

# global occupancy shape across the 30-min cycle
o = occ_ph/occ_ph.sum()
print("\nGLOBAL occupancy per minute-into-cycle (share of cycle traffic):")
print("  min :", " ".join(f"{i:4d}" for i in range(NPH)))
print("  pct :", " ".join(f"{100*v:4.1f}" for v in o))
print(f"  peak at min {int(np.argmax(o))}, trough at min {int(np.argmin(o))}, "
      f"peak/trough ratio {o.max()/o.min():.1f}")

# busiest two-way cells: both x-directions well populated
tot = xpos.sum(1)+xneg.sum(1)+ypos.sum(1)+yneg.sum(1)
twoway = (np.minimum(xpos.sum(1),xneg.sum(1)) / np.maximum(tot,1)) > 0.15
cand = [c for c in np.argsort(tot)[::-1] if twoway[c]][:4]
print("\nTWO-WAY cells: +x share of along-platform movers, per minute-into-cycle")
print("  (a swing = boarding one way, then alighting the other, around the train)")
for c in cand:
    fx_share = xpos[c]/np.maximum(xpos[c]+xneg[c],1)
    gx,gy = c//ny, c%ny
    xm,ym = XB[0]+gx*VS, YB[0]+gy*VS
    print(f"  cell {c} (x~{xm:+.0f}m y~{ym:+.0f}m, {int(tot[c]):,} movers): "
          + " ".join(f"{100*v:3.0f}" for v in fx_share))
