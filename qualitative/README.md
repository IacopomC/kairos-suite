# Qualitative panels

What a trained flow state looks like over a floor plan, read out hour by hour
on one day, in a few named regions.

- `export_flow_mod.py` — asks a frozen `flow_state.bin` for the mapped cells of
  each region at one instant per hour and writes each answer as a map of
  dynamics (one row per mixture component). Needs the bindings, so it runs in
  the pipeline container.
- `plot_mod_panels.py` — draws those files: the circular-mean field and every
  mixture component, over the dataset's occupancy grid. Needs numpy,
  matplotlib and pillow only.
- `run_qualitative_atc.sh` — both steps for one ATC state.

```
STATE=results/atc/train/flow_state.bin \
DATE=2013-09-01 REGIONS="open=-20,0,-10,10 corridor=15,25,-17,-7" \
  qualitative/run_qualitative_atc.sh      # from the suite checkout
```

`CONFIG` must be the config the state was trained under.
Regions and the drawing pitch are arguments.
