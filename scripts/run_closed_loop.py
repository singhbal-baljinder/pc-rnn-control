import numpy as np
import torch
import torch.nn as nn
from pc_rnn_control.robot.motors import (
    Spiraling_tentacle_actuation,
)
from pc_rnn_control.robot.policy import load_checkpoint
from pc_rnn_control.robot.frames import (
    normalize_it,
    un_normalize_it,
    create_reference_frame,
    change_reference_frame_dataset,
)
from pc_rnn_control.methods.predictive_coding import MyTemporalPC
from pc_rnn_control.methods.normalization import exponential_smoothing
from time import *
import signal
import sys
import os
import matplotlib.pyplot as plt
from pc_rnn_control.methods.online_learning import OnlineLearningModel

METERS_TO_MILLIMETERS = 1000
MILLIMETERS_TO_METERS = 1e-3


# Import NatNetClient
from pc_rnn_control.mocap.natnet_client import *

# ---------------------------------------------------------------------------------#
#               INITIALIZE EXPERIMETNT RELATED VARIABLES
#       (OPTITRACK, CONTROL BOX CONNECTION AND BABBLING DATA LOADING )
# ---------------------------------------------------------------------------------#
# Camera collection function def
positions = {}
quaternions = {}


def receive_rigid_body_frame(new_id, position, rotation):
    # global positions
    positions[new_id] = np.array(position)
    quaternions[new_id] = np.array(rotation)
    # print( "Received frame for rigid body", new_id," ",position )


# get markers position
def receive_marker_frame(marker_list):
    for i, marker in enumerate(marker_list):
        print(f"Marker {i} : x = {marker[0]:.3f}, y={marker[1]:.3f}, z={marker[2]:.3f}")


def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len = len(arg_list)
    if arg_list_len > 1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len > 2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len > 3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False
    return args_dict


# --------------------------------------------------------------------------------- #
# ----------------------------- LSTM model loading ------------------------------- #
# --------------------------------------------------------------------------------- #
# Get absolute path of the current script
current_script_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_script_path)
save_path = os.path.join(current_folder_path, "..", "data", "closed_loop")
# Try loading a trained model from the repo's learned_models directory (if present)
models_dir = save_path
ckpt_path = os.path.join(models_dir, f"lstm_inverse_dynamics.pth")
loaded_model = None
device = "cpu"
if os.path.exists(ckpt_path):
    ckpt_params = {
        "input_dim": 15,
        "action_dim": 3,
        "device": device,
    }
    loaded_model = load_checkpoint(ckpt_path, ckpt_params)
else:
    print(f"No checkpoint found at {ckpt_path}. Exiting.")
    exit(1)
index_obs_frames = [1, 3, 5, 7]
indices_obs_pos = [i for k in index_obs_frames for i in range(3 * k, 3 * k + 3)]
indices_obs_quat = [i for k in index_obs_frames for i in range(4 * k, 4 * k + 4)]
# --------------------------------------------------------------------------------- #
# ----------------------------- Load babbling data------------------------------- #
# --------------------------------------------------------------------------------- #
# Load actuations data dynamically_saturating_actuations
version = 35
desired_freq_babbling = [10]
idx_trial = 0
# Load the reference trajectory this run tracks
babbling_done_act_path = os.path.join(
    save_path,
    f"spiraling_babbling_{desired_freq_babbling[0]}_Hz_test_trajectory_v{version}_mvel0.5.npz",
)
executed_babbling_data = np.load(babbling_done_act_path)
# Get the robot pos from already executed babbling
observed_positions_during_babbling = executed_babbling_data["pos_array"]
observed_quaternions_during_babbling = executed_babbling_data["quat_array"]
motor_goals = executed_babbling_data["act_array"]
num_samples = observed_positions_during_babbling.shape[0]
TIME_INTERVAL_BLOCKS = [(400, 900)]
TEST_CLOSED_LOOP_WITHOUT_ADAPTATION = False
USE_ADAPTIVE_MODEL = True
ADAPTIVE_METHOD = "rnn-uoro"
condition = ""
desired_freq = [20]
if USE_ADAPTIVE_MODEL:
    file_name = f"spiraling_babbling_{int(desired_freq[0])}_Hz_test_trajectory_v{version}_mvel0.5_seq"
    file_name = file_name + ADAPTIVE_METHOD + "_adaptive"
else:
    if TEST_CLOSED_LOOP_WITHOUT_ADAPTATION:
        file_name = f"spiraling_babbling_{int(desired_freq[0])}_Hz_test_trajectory_v{version}_mvel0.5_closed_loop_seq"
    else:
        file_name = f"spiraling_babbling_{int(desired_freq[0])}_Hz_test_trajectory_v{version}_mvel0.5_open_loop_seq"
file_path = os.path.join(save_path, file_name + f"{idx_trial}" + condition + ".txt")
file_path_npz = os.path.join(save_path, file_name + f"{idx_trial}" + condition + ".npz")
# Translate data to the reference frame of first rigid-body at the beginning of the experiment
G_p_T1 = observed_positions_during_babbling[0, 0:3]
G_q_T1 = observed_quaternions_during_babbling[0, 0:4]
G_T_T1 = create_reference_frame(G_p_T1, G_q_T1)
T1_pos_Ti, _ = change_reference_frame_dataset(
    observed_positions_during_babbling, observed_quaternions_during_babbling, G_T_T1
)
# Get min and max
P_MIN = loaded_model.pos_min
P_MAX = loaded_model.pos_max
A_MIN = loaded_model.act_min
A_MAX = loaded_model.act_max
observed_positions_during_babbling_n, _, _ = normalize_it(
    T1_pos_Ti[:, indices_obs_pos], P_MIN, P_MAX
)
actions_during_babbling_n, _, _ = normalize_it(
    executed_babbling_data["act_array"], A_MIN, A_MAX
)

# --------------------------------------------------------------------------------- #
# ----------------------------- spiraling class start ------------------------------- #
# --------------------------------------------------------------------------------- #
dynamixel_port = "/dev/ttyUSB0"
print("Connected COM ports: " + dynamixel_port)
N_motors = 3
VEL_UNIT = 0.229
spiraling_operation = Spiraling_tentacle_actuation(
    N_MOTORS=N_motors, DEVICENAME=dynamixel_port
)

# Scan and print the connected DYNAMIXEL IDs
spiraling_operation.scan_and_print_devices()
# Port is opened in the initialization of the class instant
# Baudrate is set in the initialization of the class instant
spiraling_operation.torque_enable()

# now set desired initial positions for the motors
# (this value could be set as the Homing position in the Non volatile memory of the Dynamixels)
desired_init_motor_pos = motor_goals[0, :].astype(int)
spiraling_operation.reset_dxl = desired_init_motor_pos
spiraling_operation.reset_dxl_positions()
print("Robot class ready...")
read_motor = spiraling_operation.read_dxl_position()
print("Initial motor positions after reset: ", read_motor)
current_motor_angles_n, _, _ = normalize_it(
    read_motor.reshape(1, -1).copy(), A_MIN, A_MAX
)
# Set min and max position for each motor
min_array = np.asarray([400, 1000, 200], dtype=int)
max_array = np.asarray([8000, 8000, 8000], dtype=int)

# Actuation_frequency
actuation_sampling_time = 1 / desired_freq[0]
coeff = 0.5
motor_frequency = desired_freq[0] * coeff  # Hz
custom_velocity = np.array(motor_frequency / VEL_UNIT * np.ones(N_motors), dtype="int")
print("Desired velocity: ", custom_velocity)

goal_position = motor_goals[0, :]
goal_cmd = motor_goals[0, :]

init_motor_position = spiraling_operation.read_dxl_position()
print("Initial motor positions: ", init_motor_position)
print("Goal positions: ", goal_position)

# ---------------------------------------------------------------------------------- #
# ----------------------------- Initialize FW model- ------------------------------- #
# ---------------------------------------------------------------------------------- #
# Parameters and hyperparameters
dim_input = motor_goals.shape[1]
dim_output = len(index_obs_frames) * 3

# Initialize model using OnlineLearningModel
forward_model = OnlineLearningModel(
    model_type=ADAPTIVE_METHOD, input_dim=dim_input, output_dim=dim_output
)
fw_model_predictions = []
fw_model_observations = []
# ---------------------------------------------------------------------------------- #
# ----------------------------- Initialize variables ------------------------------- #
# ---------------------------------------------------------------------------------- #
# Initialize the mocap frames positions and motor positions for inverse dynamics model
# Inputs are of shape (1, T, input_dim) (only one batch when running in real robot)
num_past_observations = 3  # including current observation (MUST BE  AT LEAST 2)
num_position_states = len(index_obs_frames) * 3
num_orientation_states = len(index_obs_frames) * 4
curr_position_n = observed_positions_during_babbling_n[1, :]
input_pos = torch.zeros(
    (1, num_past_observations, num_position_states), dtype=torch.float32
).to(device)
input_pos[:, -1, :] = torch.tensor(
    observed_positions_during_babbling_n[1, :], dtype=torch.float32
).to(device)
input_act = torch.zeros(
    (1, num_past_observations, loaded_model.action_dim), dtype=torch.float32
).to(device)

input_act[:, :, :] = torch.tensor(
    current_motor_angles_n.reshape(-1), dtype=torch.float32
).to(device)
input = torch.concatenate([input_pos, input_act], axis=-1).to(device)
# Outputs are of shape (1, T, action_dim)
output = torch.zeros(
    (1, num_past_observations, loaded_model.action_dim), dtype=torch.float32
).to(device)
pos = observed_positions_during_babbling[0, :]
num_past_positions_as_input = 3
num_past_actions_as_input = 3

last_act_n = np.zeros(
    dim_input,
)
# --------------------------------------------------------------------------------- #
# ----------------------------- Main control loop ------------------------------- # #
# --------------------------------------------------------------------------------- #
if __name__ == "__main__":

    optionsDict = {}
    # Init optitrack param
    optionsDict["clientAddress"] = "169.254.164.224"
    optionsDict["serverAddress"] = "169.254.164.223"
    optionsDict["use_multicast"] = False
    optionsDict = my_parse_args(sys.argv, optionsDict)

    # This will create a new NatNet client
    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])
    streaming_client.run()
    streaming_client.rigid_body_listener = receive_rigid_body_frame
    # Sleep to make sure the connection with Optitrack is finished
    time.sleep(0.01)

    time_send_cmd = []
    time_receive_obs_and_save = []
    time_predictions_and_update = []
    fixed_model_inference_time = []
    total_time_cycle = []

    goals_tosave = []
    pos_tosave = []
    quar_tosave = []
    motor_pos_tosave = []
    predictions_tosave = []
    sample_time_idx = 0
    try:
        with open(file_path, "w") as f:
            for itr_n in range(0, num_samples - 1):

                time_total0 = time.time()

                # ------- actuate the muscle
                time_send_cmd0 = time.time()

                # ---------------------------------------------------------------- #
                # ------------------ Update input to controller ------------------ #
                # ---------------------------------------------------------------- #
                error_pred_fw = forward_model.predict(last_act_n)
                time_predictions_and_update.append(time.time() - time_send_cmd0)
                fw_model_predictions.append(error_pred_fw)

                if np.isnan(error_pred_fw).any():
                    print(f"NaN detected in error_pred_fw.")
                    error_pred_fw = np.zeros_like(error_pred_fw)
                # Populate the input postions by shifting previous observations and adding the current one
                input_pos[:, :-1, :] = input_pos[:, 1:, :].clone()
                if num_past_positions_as_input is not None:
                    input_pos[:, : -num_past_positions_as_input - 1, :] = 0.0
                # Set desired target position
                target_position_n = torch.tensor(
                    observed_positions_during_babbling_n[itr_n + 1, :],
                    dtype=torch.float32,
                ).to(device)

                if USE_ADAPTIVE_MODEL:
                    correction_n = target_position_n + torch.tensor(
                        error_pred_fw, dtype=torch.float32
                    ).to(device)
                else:
                    correction_n = target_position_n

                input_pos[:, -1, :] = correction_n
                if TEST_CLOSED_LOOP_WITHOUT_ADAPTATION:
                    input_pos[:, -2, :] = torch.tensor(curr_position_n, dtype=torch.float32).to(device)
                print(f"Control loop it {sample_time_idx}")
                # Populate the input actuation: the model is assumed to know only the previous action
                input_act[:, :-1, :] = input_act[:, 1:, :].clone()
                if num_past_actions_as_input is not None:
                    input_act[:, : -num_past_actions_as_input - 1, :] = 0.0
                # Prepare actuation input for next iteration
                current_motor_angles_n, _, _ = normalize_it(
                    read_motor.reshape(1, -1).copy(), A_MIN, A_MAX
                )
                input_act[:, -1, :] = torch.tensor(
                    current_motor_angles_n.reshape(-1), dtype=torch.float32
                ).to(device)
                # input_act[:, -1, :] = output[0, -1, :].clone()
                input = torch.concatenate([input_pos, input_act], axis=-1).to(device)

                # -------------------------------------------------------------------------------------------------- #
                # ----------------------------- Compute control and sent it to robot ------------------------------- #
                # -------------------------------------------------------------------------------------------------- #
                # Get predicted torque
                time_before_fixed_model_inference = time.time()
                output = loaded_model(input)
                fixed_model_inference_time.append(time.time() - time_before_fixed_model_inference)
                # Denormalize and approx as int before sending to low level motor controllers
                goal_position = un_normalize_it(
                    output[0, -1, :].cpu().detach().numpy().reshape(1, -1), A_MIN, A_MAX
                )
                predictions_tosave.append(goal_position.astype(int)[0, :])
                # Make sure the goal positions are within the limits
                goal_position = np.clip(goal_position[0, :], min_array, max_array)
                
                # Gated on the run's condition tag: `_fault` is the frozen-actuator
                # condition, "" the nominal one. Without the gate the block below
                # runs in both, and `condition` only renames the output file.
                is_jammed = bool(condition) and any(
                    start < itr_n < end for start, end in TIME_INTERVAL_BLOCKS
                )

                if is_jammed:
                    # Freeze motor 0 by NOT updating it: its goal keeps the value
                    # it held when the fault window opened, while the healthy
                    # motors 1 and 2 keep following the controller.
                    healthy_idx = [1, 2]
                    # Keep the previous goal_cmd for indices 1 & 2, only update index 0
                    goal_cmd[healthy_idx] = goal_position[healthy_idx].astype(int)
                    print(f"Iteration {itr_n}: Actuator Jammed! Updating only motor {healthy_idx}")
                else:
                    # 3. Normal operation: All motors follow the controller
                    goal_cmd = goal_position.astype(int)
                
                spiraling_operation.write_dxl_goal_velocity(
                    vel=custom_velocity.astype(int), pos=goal_cmd
                )
                time_send_cmd.append(time.time() - time_send_cmd0)

                # --------------------------------------------------------------------------------- #
                # ----------------------------- Sense and save data ------------------------------- #
                # --------------------------------------------------------------------------------- #
                # Forword model prediction (before any position is received)
                time_receive_obs_and_save0 = time.time()

                # Read all rigid bodies (populated by a call back)
                pos_cur = positions
                quat_curr = quaternions

                # Parse the optitrack readings (assuming 5 rigid bodies are sensed)
                pos_list = [np.zeros((3,)) for _ in range(8)]
                for rb_id in pos_cur.keys():
                    pos_list[rb_id - 1] = pos_cur[rb_id]

                pos = np.vstack(pos_list)

                # Save the sample index, the commanded motor goals and the
                # measured marker positions
                f.write(str([sample_time_idx, goal_cmd, pos]).strip("[]"))
                f.write("\n")
                sample_time_idx += 1

                quar_list = [np.zeros((4,)) for _ in range(8)]
                for rb_id in pos_cur.keys():
                    quar_list[rb_id - 1] = quat_curr[rb_id]

                # Update lists to be saved in a npz file
                goals_tosave.append(goal_cmd)
                pos_temp_matrix = np.hstack(pos_list)
                pos_tosave.append(pos_temp_matrix)
                quat_temp_matrix = np.hstack(quar_list)
                quar_tosave.append(quat_temp_matrix)
                # Save motors encoder position
                read_motor = spiraling_operation.read_dxl_position()
                motor_pos_tosave.append(read_motor)
                time_receive_obs_and_save.append(
                    time.time() - time_receive_obs_and_save0
                )

                # -------------------------------------------------------------------------- #
                # ------------------ Learn FW model ---------------------------------------- #
                # -------------------------------------------------------------------------- #
                # Prepare inputs and predictions for the forward model

                last_act_n, _, _ = normalize_it(
                    goal_cmd.reshape(1, -1),
                    A_MIN,
                    A_MAX,
                )
                # Change reference frame to top robot frame at the beginning of the experiment
                # Translate data to the reference frame of first rigid-body at the beginning of the experiment
                if itr_n == 0:
                    G_p_T1 = pos_temp_matrix[0:3]
                    G_q_T1 = quat_temp_matrix[0:4]
                    G_T_T1 = create_reference_frame(G_p_T1, G_q_T1)

                T1_pos_Ti_curr, _ = change_reference_frame_dataset(
                    pos_temp_matrix.reshape(1, -1),
                    quat_temp_matrix.reshape(1, -1),
                    G_T_T1,
                )

                curr_position_n, _, _ = normalize_it(
                    T1_pos_Ti_curr[0, indices_obs_pos].reshape(1, -1), P_MIN, P_MAX
                )

                #### NOTA BENE: the observation of the FW model is the tracking error
                target_pose_n_np = observed_positions_during_babbling_n[itr_n + 1, :]
                time_before_update = time.time()
                forward_model.update(last_act_n, target_pose_n_np - curr_position_n[0])
                time_predictions_and_update[-1] += time.time() - time_before_update
                fw_model_observations.append(target_pose_n_np - curr_position_n[0])


                # ------------------------------------------- #
                # ------------------ Sleep ------------------ #
                # ------------------------------------------- #
                # --- Sleep to maintain frequency ---
                elapsed_time = time.time() - time_total0
                time_to_sleep = actuation_sampling_time - elapsed_time

                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else:
                    # Optional: Log a warning if the computations took longer than the 100ms budget
                    print(f"Warning: Loop overrun! Cycle took {elapsed_time*1000:.2f}ms")

                total_time_cycle.append(time.time() - time_total0)


    except KeyboardInterrupt:
        # Close file if it was opened
        if not f.closed:
            f.close()
        # Sleep for 2 seconds before exiting
        time.sleep(2.0)
        # Set motor to zero
        for _ in range(2):
            spiraling_operation.reset_dxl_positions()
            sleep(3.0)
            spiraling_operation.torque_disable()

        ## Convert to array for an alternative save format that contains also the orientations
        act_list = np.array(goals_tosave)
        pos_list = np.array(pos_tosave)
        quat_list = np.array(quar_tosave)
        motor_pos_list = np.array(motor_pos_tosave)
        total_time_cycle = np.array(total_time_cycle)
        predictions_tosave = np.array(predictions_tosave)
        fw_model_observations = np.array(fw_model_observations)
        fw_model_predictions = np.array(fw_model_predictions)
        time_predictions_and_update = np.array(time_predictions_and_update)
        fixed_model_inference_time = np.array(fixed_model_inference_time)

        # Save to file
        np.savez(
            file_path_npz,
            act_array=act_list,
            pos_array=pos_list,
            quat_array=quat_list,
            total_time_cycle=total_time_cycle,
            motor_pos=motor_pos_list,
            pred_array=predictions_tosave,
            fw_model_observations=fw_model_observations,
            fw_model_predictions=fw_model_predictions,
            time_predictions_and_update=time_predictions_and_update,
            fixed_model_inference_time=fixed_model_inference_time
        )
        print("Finished.")

        exit()

    # Sleep for 2 seconds before exiting
    time.sleep(2.0)
    # Set motor to zero
    for _ in range(2):
        spiraling_operation.reset_dxl_positions()
        sleep(3.0)
        spiraling_operation.torque_disable()

    ## Convert to array for an alternative save format that contains also the orientations
    act_list = np.array(goals_tosave)
    pos_list = np.array(pos_tosave)
    quat_list = np.array(quar_tosave)
    motor_pos_list = np.array(motor_pos_tosave)
    total_time_cycle = np.array(total_time_cycle)
    predictions_tosave = np.array(predictions_tosave)
    fw_model_observations = np.array(fw_model_observations)
    fw_model_predictions = np.array(fw_model_predictions)
    time_predictions_and_update = np.array(time_predictions_and_update)
    fixed_model_inference_time = np.array(fixed_model_inference_time)
    # Save to file
    np.savez(
        file_path_npz,
        act_array=act_list,
        pos_array=pos_list,
        quat_array=quat_list,
        total_time_cycle=total_time_cycle,
        motor_pos=motor_pos_list,
        pred_array=predictions_tosave,
        fw_model_observations=fw_model_observations,
        fw_model_predictions=fw_model_predictions,
        time_predictions_and_update=time_predictions_and_update,
        fixed_model_inference_time=fixed_model_inference_time
    )
    print("Finished.")

    exit()
