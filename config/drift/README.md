# Simulated odometric drift

One file per run of the loop-closure study. `train_atc.py --drift-config <file>`
reads the drift model from here; `--drift-per-loop-m` and
`--drift-yaw-per-loop-deg` override single entries for a one-off sweep.

## Entries

| key | meaning |
| --- | --- |
| `per_loop_m` | translational error accumulated per completed patrol loop, in metres. Grows linearly with distance travelled, along a direction that wanders; resets at each correction. |
| `yaw_per_loop_deg` | heading error accumulated per completed patrol loop, in degrees. Grows the same way and resets with the translation. |
| `seed` | seeds the wandering direction, so a run replays identically. |
| `control_spacing_s` | spacing of the warp control points along the trajectory. |

Both rates are zero in the drift-free run, which fires no loop closures and
delivers no correction warp.

## Where the rates come from

Both are a fraction of distance travelled, taken from what odometry systems
report. Translation is quoted as a percentage of distance: on the order of
$0.5\%$ for visual-inertial and $0.1\%$ for LiDAR-inertial systems. Rotation is
quoted in deg/m: the KITTI odometry benchmark lists $0.0013$ deg/m for the
leading LiDAR methods (LOAM, V-LOAM) and $0.0033$ deg/m for VINS-Fusion, whose
translation error is $1.09\%$.

Each run applies the same fraction to both channels:

| run | translation | rotation |
| --- | --- | --- |
| loop closure | $0.09\%$, just under the LiDAR-inertial figure | $0.00117$ deg/m, $90\%$ of the LiDAR figure |
| uncorrected | $0.005\%$, twenty times lower | $0.000065$ deg/m, twenty times lower |

Per-distance rates transfer between datasets; per-loop numbers depend on the
loop length, 659 m on ATC and 158 m on HB. The `yaw_per_loop_deg` entry is
the per-distance rate multiplied by that dataset's loop length.

## Reading a yaw rate as a displacement

A yaw error turns everything the platform registers about its own position, so
it displaces a mapped point in proportion to its distance from the trajectory:
a yaw of `psi` displaces a point at lever arm `L` by `psi * L`. The ATC graph
spans 93 x 51 m and reaches 54.4 m from its centroid; the HB graph reaches
19.4 m. On ATC that makes 0.21 deg half a voxel at the far side and 0.42 deg a
full voxel, so the uncorrected run's 4.2 deg over a day displaces the far side
by 4.0 m against the 3.0 m its translation contributes.
