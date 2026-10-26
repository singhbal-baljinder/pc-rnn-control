"""
Robot babbling-data loader.

Extracted from the comprehensive online-learning comparison script:
loads the spiraling_robot (NPZ) dataset,
normalises actions and positions, and exposes them as a clean dataset
object.

Depends on the normalisation helpers in `normalization`.
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

from pc_rnn_control.methods.normalization import (
    normalize_action_data,
    normalize_position_data,
    denormalize_position_data,
)

MILLIMETERS_TO_METERS = 1e-3
METERS_TO_MILLIMETERS = 1000.0


@dataclass
class RobotDataset:
    """Container for one trajectory of normalised actions/observations."""
    X:        np.ndarray        # (T, n_u)  normalised motor commands
    y:        np.ndarray        # (T, n_y)  normalised end-effector positions
    y_un:     np.ndarray        # (T, n_y)  un-normalised positions [m]
    pos_min:  np.ndarray
    pos_max:  np.ndarray
    freq:     float
    robot_name: str
    filename:   str

    @property
    def n_u(self) -> int: return self.X.shape[1]

    @property
    def n_y(self) -> int: return self.y.shape[1]

    @property
    def T(self) -> int:   return self.X.shape[0]

    def denormalize(self, y_n: np.ndarray) -> np.ndarray:
        """Take a normalised prediction back to metres."""
        return denormalize_position_data(y_n, self.pos_min, self.pos_max)


def load_robot_data(robot_name: str,
                    data_dir:   str,
                    filename:   str,
                    n_samples:  Optional[int] = 10_000,
                    freq:       float = 40.0) -> RobotDataset:
    """
    Load a recorded babbling or test trajectory as a RobotDataset.

    Args:
        robot_name : 'spiraling_robot'
        data_dir   : directory containing the dataset file
        filename   : dataset stem WITHOUT extension
        n_samples  : truncate to first n_samples timesteps (None = all)
        freq       : sampling frequency in Hz, recorded on the dataset
    """
    if robot_name == "spiraling_robot":
        path = os.path.join(data_dir, filename + ".npz")
        results_exp = np.load(path)
        positions_n, pos_min, pos_max = normalize_position_data(
            results_exp["pos_array"])
        actions_n, _, _ = normalize_action_data(results_exp["act_array"])
        positions_un = results_exp["pos_array"]

    else:
        raise ValueError(f"Unknown robot_name: {robot_name!r}")

    if n_samples is not None:
        actions_n    = actions_n[:n_samples]
        positions_n  = positions_n[:n_samples]
        positions_un = positions_un[:n_samples]

    return RobotDataset(
        X=actions_n, y=positions_n, y_un=positions_un,
        pos_min=pos_min, pos_max=pos_max, freq=freq,
        robot_name=robot_name, filename=filename,
    )




