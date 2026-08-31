from scipy.signal import savgol_filter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import torch
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
import os


METERS_TO_MILLIMETERS = 1000
MILLIMETERS_TO_METERS = 1e-3




def normalize_position_data(robot_positions_un, min_val=None, max_val=None):
    """
    Normalize robot position data
    """

    if min_val is None and max_val is None:
        total_min = np.zeros(robot_positions_un.shape[1])
        total_max = np.zeros(robot_positions_un.shape[1])
        for i in range(robot_positions_un.shape[1]):
            total_min[i] = round(min(robot_positions_un[:, i]), 5)
            total_max[i] = round(max(robot_positions_un[:, i]), 5)

        total_min, total_max = np.array(total_min), np.array(total_max)
    else:
        total_min = min_val
        total_max = max_val

    # Normalize the data
    robot_positions_n = np.zeros(robot_positions_un.shape)
    scaling_factor = 1.0
    for i in range(robot_positions_un.shape[1]):
        robot_positions_n[:, i] = (robot_positions_un[:, i] - total_min[i]) / (
            total_max[i] - total_min[i]
        )
        robot_positions_n[:, i] = robot_positions_n[:, i] * scaling_factor
        robot_positions_n[:, i] = np.round(robot_positions_n[:, i], 5)

    return robot_positions_n, total_min, total_max


def normalize_action_data(actions_un, min_val=None, max_val=None):
    """
    Normalize actuation data to [0, 1] using per-channel min/max.
    """
    if min_val is None and max_val is None:
        total_min = np.zeros(actions_un.shape[1])
        total_max = np.zeros(actions_un.shape[1])
        for i in range(actions_un.shape[1]):
            total_min[i] = round(min(actions_un[:, i]), 5)
            total_max[i] = round(max(actions_un[:, i]), 5)
        total_min, total_max = np.array(total_min), np.array(total_max)
    else:
        total_min = min_val
        total_max = max_val

    scaling_factor = 1.0
    actions_n = np.zeros(actions_un.shape)
    for i in range(actions_un.shape[1]):
        actions_n[:, i] = (actions_un[:, i] - total_min[i]) / (
            total_max[i] - total_min[i]
        )
        actions_n[:, i] = actions_n[:, i] * scaling_factor
        actions_n[:, i] = np.round(actions_n[:, i], 5)

    return actions_n, total_min, total_max


def denormalize_position_data(robot_positions_n, total_min, total_max):
    """
    Denormalize robot position data
    """
    scaling_factor = 1.0
    robot_positions_un = np.zeros(robot_positions_n.shape)
    if len(robot_positions_n.shape) == 2:
      for i in range(robot_positions_n.shape[1]):
          robot_positions_un[:, i] = (robot_positions_n[:, i] / scaling_factor) * (
              total_max[i] - total_min[i]
          ) + total_min[i]
          robot_positions_un[:, i] = np.round(robot_positions_un[:, i], 5)
    elif len(robot_positions_n.shape) == 1:
      for i in range(robot_positions_n.shape[0]):
        robot_positions_un[i] = (robot_positions_n[i] / scaling_factor) * (
            total_max[i] - total_min[i]
        ) + total_min[i]
        robot_positions_un[i] = np.round(robot_positions_un[i], 5)

    return robot_positions_un










def exponential_smoothing(data, alpha=0.0):
    """
    Apply exponential smoothing to the data.
    This function smooths the data using a simple exponential smoothing technique.
    """
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # First value is the same
    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
    return smoothed_data



# Variant: also returns Savitzky-Golay filtered derivative of y
























from dataclasses import dataclass, field






def create_full_rank_matrix(rows, cols):
    while True:
        matrix = torch.rand(rows, cols) / max(rows, cols)
        if torch.linalg.matrix_rank(matrix) == min(rows, cols):
            return matrix
