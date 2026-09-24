#!/usr/bin/env python3
"""TBD robot-dataset pedestrian stream (one scene's tracks.csv).

Columns: t_ns, ped_id, x_world, y_world, z_world, theta_rad, rho_mps. Heading and
speed are already provided by the tracker, so nothing is derived. Exposes the same
interface as AtcStream/EindhovenStream: t (s, absolute unix), id, x/y (m), theta
(rad), rho (m/s). Each scene is a ~25 min robot recording; scenes share one world
frame, so pooling them god's-eye is valid, and their absolute timestamps let a
short-period fold align every scene to the same cycle grid.
"""
import numpy as np
import pandas as pd


class TbdStream:
    def __init__(self, path, x_bounds=None, y_bounds=None):
        df = pd.read_csv(path, usecols=["t_ns", "ped_id", "x_world", "y_world",
                                        "theta_rad", "rho_mps"])
        self.t = df["t_ns"].to_numpy(dtype=np.float64) / 1e9
        self.id = df["ped_id"].to_numpy()
        self.x = df["x_world"].to_numpy(dtype=np.float64)
        self.y = df["y_world"].to_numpy(dtype=np.float64)
        self.theta = df["theta_rad"].to_numpy(dtype=np.float64)
        self.rho = df["rho_mps"].to_numpy(dtype=np.float64)
        # AtcStream sorts by time and callers rely on it: the FoV extractor
        # slices each keyframe window with np.searchsorted, which is only valid
        # on a monotonic self.t.
        order = np.argsort(self.t, kind="stable")
        self.t = self.t[order]; self.id = self.id[order]; self.x = self.x[order]
        self.y = self.y[order]; self.theta = self.theta[order]
        self.rho = self.rho[order]
        if x_bounds and y_bounds:
            m = ((self.x >= x_bounds[0]) & (self.x < x_bounds[1]) &
                 (self.y >= y_bounds[0]) & (self.y < y_bounds[1]))
            self.t = self.t[m]; self.id = self.id[m]; self.x = self.x[m]
            self.y = self.y[m]; self.theta = self.theta[m]; self.rho = self.rho[m]
        self.t_min = float(self.t[0]) if len(self.t) else 0.0
        self.t_max = float(self.t[-1]) if len(self.t) else 0.0


class TbdPoseTrack:
    """Recorded robot trajectory for one TBD scene (`poses.csv`).

    TBD is a real robot recording, so the embodied field of view is anchored to
    the logged pose rather than to the simulated patrol `robot_sim.RobotSim`
    synthesises for ATC / HB. Columns: timestamp_ns, px, py, pz, qw, qx, qy, qz.
    Only the planar position is used, matching the radial active-window gate in
    FlowTemporalModule::processFrame.
    """

    def __init__(self, path):
        df = pd.read_csv(path, usecols=["timestamp_ns", "px", "py"])
        t = df["timestamp_ns"].to_numpy(dtype=np.float64) / 1e9
        order = np.argsort(t, kind="stable")
        self.t = t[order]
        self.x = df["px"].to_numpy(dtype=np.float64)[order]
        self.y = df["py"].to_numpy(dtype=np.float64)[order]
        self.t_min = float(self.t[0])
        self.t_max = float(self.t[-1])

    def pose_at(self, t_s):
        """Planar pose at absolute time `t_s`, clamped to the recorded span."""
        return (float(np.interp(t_s, self.t, self.x)),
                float(np.interp(t_s, self.t, self.y)))
