"""LSTM inverse-dynamics policy for the cable-driven arm.

Trained once on babbling data by scripts/train_policy.py and then held FIXED:
the online learners in the paper adapt around this policy rather than
modifying it. Training recordings are .npz with `pos_array` (marker
positions) and `motor_pos` (executed motor commands).
"""

from __future__ import annotations
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from .frames import *





class LSTMInverseDynamics(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_elman_rnn: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.pos_min = None
        self.pos_max = None
        self.act_min = None
        self.act_max = None
        self.use_elman = use_elman_rnn
        # LSTM expects input (T, B, input_size) if batch_first=False; we'll use batch_first=True
        # pass dropout to the LSTM (applies between layers) and also expose an external
        # dropout layer applied to the LSTM outputs before the head for extra regularization
        if use_elman_rnn:
            self.recurrent_layer = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )

        else:
            self.recurrent_layer = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        self.layer_1 = nn.Linear(hidden_size, action_dim, bias=False)

        self.layer_2 = nn.Sigmoid()

    def forward(self, x):
        # x: (B, T, input_dim)
        s, _ = self.recurrent_layer(x)
        l1 = self.layer_1(s)

        # actions: (B, T, action_dim)
        actions = self.layer_2(l1)
        return actions


class InverseDynamicsWindowDataset(Dataset):
    """Dataset that loads a babbling NPZ (or states/actions NPZ) and produces
    fixed-length windows of past states and actions to predict the current action.

            Each sample is a pair (input_window, target_action) where
              - input_window has shape (K, input_dim + action_dim) and contains per-timestep
                concatenated [state, action] features. The last action in the window is
                zeroed out so the model cannot peek the target.
              - target_action has shape (action_dim,) and is the action at the last timestep
                of the window.

            Parameters
            - path: path to npz file (supports 'pos_array'/'act_array' or 'states'/'actions')
            - seq_len: window length K (required)
            - stride: hop between windows (default = seq_len for non-overlapping)
            - version: different versions define different input/output formats
    """

    def __init__(
        self,
        path: str,
        seq_len: int,
        num_past_actions_as_input: int = None,
        num_past_positions_as_input: int = None,
        use_sequence: bool = False,
        pos_min=None,
        pos_max=None,
        action_min=None,
        action_max=None,
        plot_dataset=False,
        index_obs_frames=None,
        learn_with_pose_error=False,
    ):
        if seq_len is None or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if use_sequence:
            self.seq_len = int(seq_len)
        else:
            self.seq_len = 5
        if self.seq_len <= 2:
            raise ValueError("seq_len must be greater than 2 to allow zeroing actions")
        if index_obs_frames is None:
            raise ValueError(
                "index_obs_frames must be a list of integers corresponding to the optitrack frame indeces to consider for training"
            )
        indices_obs_pos = [i for k in index_obs_frames for i in range(3 * k, 3 * k + 3)]
        indices_obs_quat = [
            i for k in index_obs_frames for i in range(4 * k, 4 * k + 4)
        ]

        data = np.load(path)
        # load either format
        if "pos_array" in data and "motor_pos" in data:
            pos = np.asarray(data["pos_array"]).astype(np.float32)
            quat = np.asarray(data["quat_array"]).astype(np.float32)
            act = np.asarray(data["motor_pos"]).astype(np.float32)
        else:
            raise ValueError("npz must contain 'pos_array' and 'motor_pos'")
        # Before training we need to change the reference to the one at the top of the arm at the beginning of the experiment
        # Get the pose of the frame of the first set of markers at the beginning of the experiment
        G_p_T1 = pos[0, 0:3]
        G_q_T1 = quat[0, 0:4]
        G_T_T1 = create_reference_frame(G_p_T1, G_q_T1)
        T1_pos_Ti, _ = change_reference_frame_dataset(
            pos[:, indices_obs_pos], quat[:, indices_obs_quat], G_T_T1
        )

        # Get global min and max for the action
        # action_min = [np.min(act) for _ in range(0, act.shape[1])]
        # action_max = [np.max(act) for _ in range(0, act.shape[1])]
        # Convert positions from meters to millimeters
        if pos_min is not None and pos_max is not None:
            states, _, _ = normalize_it(
                T1_pos_Ti, total_p_min=pos_min, total_p_max=pos_max
            )
        else:
            states, pos_min, pos_max = normalize_it(T1_pos_Ti)
        if action_min is not None and action_max is not None:
            actions, _, _ = normalize_it(
                act,
                total_p_min=action_min,
                total_p_max=action_max,
                use_percentage_change_limit=True,
            )
        else:
            actions, action_min, action_max = normalize_it(
                act, use_percentage_change_limit=True
            )

        if plot_dataset:
            # Visualize dataset
            visualize_3D_points_scatter(states)
        self.min_pos = pos_min
        self.max_pos = pos_max
        self.min_action = action_min
        self.max_action = action_max
        # Promote 2D recordings to (1, T, D)
        states = states[None, ...]
        actions = actions[None, ...]

        # Plot loaded data for verification
        if plt is not None:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(states[0, :, :])
            ax1.set_title("Loaded States")
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(actions[0, :, :])
            ax2.set_title("Loaded Actions")
            plt.tight_layout()
            plt.show()

        if states.ndim != 3 or actions.ndim != 3:
            raise ValueError("states/actions must be 2D (T,D) or 3D (N,T,D)")

        if states.shape[0] != actions.shape[0] or states.shape[1] != actions.shape[1]:
            raise ValueError("states and actions must have matching (N,T)")

        _, num_samples, _ = states.shape

        inputs = []
        targets = []
        if use_sequence:
            # For each sequence, tile non-overlapping windows of length K
            # Each version differes only on the particular past actions provided in the input window
            for end in range(self.seq_len + 1, num_samples - 2):
                start = end - self.seq_len
                # Use past K states and current desired state
                s_win = states[0, start + 2 : end + 2, :].copy()  # (K, input_dim)
                # Shift back actions by one to prevent peeking at the target
                a_win = actions[0, start:end, :].copy()  # (K, action_dim)
                if learn_with_pose_error:
                    s_win[-1] = s_win[-1] - s_win[-2]
                if (
                    num_past_actions_as_input is not None
                    and num_past_actions_as_input < self.seq_len
                ):
                    # Zero out all but the last num_past_actions_as_input actions
                    a_win[: -num_past_actions_as_input - 1, :] = (
                        0.0  # zero out all but last num_past_actions_as_input actions
                    )
                if (
                    num_past_positions_as_input is not None
                    and num_past_positions_as_input < self.seq_len
                ):
                    # Zero out all but the last num_past_positions_as_input states
                    s_win[: -num_past_positions_as_input - 1, :] = (
                        0.0  # zero out all but last num_past_positions_as_input states
                    )

                # Create input by concatenating states and actions
                inp = np.concatenate([s_win, a_win], axis=-1)  # (K, state + action)

                # Predict the action that leads to the desired state at time 'end'
                tgt = actions[0, end, :].copy()
                inputs.append(inp)
                targets.append(tgt)
        else:
            # Do not use the sequence information; just create samples for each time step
            for end in range(self.seq_len, num_samples - 2):
                start = end - self.seq_len
                # state and action windows (K, D)
                s_win = states[0, start : end + 2, :].copy().astype(np.float32)
                a_win = actions[0, start : end + 2, :].copy().astype(np.float32)

                # Use a particular selection of past states/actions based on the num_past_*_as_input params
                # Always zero out the last action to prevent peeking
                inp = np.concatenate(
                    [a_win[-5, :], a_win[-4, :], a_win[-3, :], s_win[-1, :]], axis=-1
                )  # (1, state + action)
                # Add an axis for the time step
                inp = inp[np.newaxis, :]
                tgt = actions[0, end, :].copy().astype(np.float32)
                inputs.append(inp)
                targets.append(tgt)

        self.inputs = np.stack(inputs, axis=0).astype(np.float32)
        self.targets = np.stack(targets, axis=0).astype(np.float32)

        # convert to torch tensors once to avoid per-sample from_numpy overhead
        self.inputs_tensor = torch.from_numpy(self.inputs).float()
        self.targets_tensor = torch.from_numpy(self.targets).float()

    def __len__(self):
        return int(self.inputs_tensor.shape[0])

    def __getitem__(self, idx):
        return self.inputs_tensor[idx], self.targets_tensor[idx]


def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            states = batch[0].to(device)
            targets = batch[1].to(device)
            preds = model(states)
            preds_last = preds[:, -1, :]
            loss = criterion(preds_last, targets)
            total_loss += loss.item() * states.size(0)
    return total_loss / len(dataloader.dataset)


def load_checkpoint(checkpoint_path: str, params: dict):
    """Load a saved checkpoint and return a model on `device` plus the params dict.

    If `full_ds` is provided, infer input/output dims from it. Otherwise the
    checkpoint must contain enough info in 'params' to recreate the model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(
        checkpoint_path, map_location=params.get("device", "cpu"), weights_only=False
    )

    # fall back to params saved in the checkpoint (if any)
    input_dim = params.get("input_dim")
    action_dim = params.get("action_dim")

    if input_dim is None or action_dim is None:
        raise ValueError(
            "input_dim and action_dim must be provided via dataset or checkpoint params"
        )

    hidden_size = ckpt.get("params").get("hidden_size", 128)
    num_layers = ckpt.get("params").get("num_layers", 2)
    dropout = ckpt.get("params").get("dropout", 0.0)

    use_elman = ckpt.get("params").get("use_elman", False)
    model = LSTMInverseDynamics(
        input_dim=input_dim,
        action_dim=action_dim,
        use_elman_rnn=use_elman,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.act_min = ckpt.get("params").get("min_act_training")
    model.act_max = ckpt.get("params").get("max_act_training")
    model.pos_min = ckpt.get("params").get("min_pos_training")
    model.pos_max = ckpt.get("params").get("max_pos_training")
    model.load_state_dict(ckpt["model_state"])
    model.to(params.get("device", "cpu"))
    model.use_elman = use_elman
    model.eval()
    return model


