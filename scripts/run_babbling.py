import numpy as np
from pc_rnn_control.robot.motors import (
    Spiraling_tentacle_actuation,
)
from time import *
import signal
import sys
import os
import matplotlib.pyplot as plt
import copy

METERS_TO_MILLIMETERS = 1000
MILLIMETERS_TO_METERS = 1e-3
PLOT = True
# Indeces of the motors to actuate

INDECES_MOTOR_TO_ACTUATE = [0, 1, 2]

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
# ----------------------------- Spiraling class start ------------------------------- #
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

# --------------------------------------------------------------------------------- #
# ----------------------------- Load babbling data------------------------------- #
# --------------------------------------------------------------------------------- #
# Get absolute path of the current script
current_script_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_script_path)
save_path = os.path.join(current_folder_path, "..", "data", "open_loop")
# Load actuations data dynamically_saturating_actuations

version = 21
babbling_act_path = os.path.join(save_path, f"test_trajectory_v{version}.npz")
motor_goals = np.load(babbling_act_path)["trajectory"]

num_samples = motor_goals.shape[0]
desired_freq = [20]
coeff = 0.5
motor_frequency = desired_freq[0] * coeff  # Hz
custom_velocity = np.array(motor_frequency / VEL_UNIT * np.ones(N_motors), dtype="int")
print("Desired velocity: ", custom_velocity)

file_name = f"spiraling_babbling_{int(desired_freq[0])}_Hz_test_trajectory_v{version}"
file_path = os.path.join(save_path, file_name + f"_mvel{coeff}.txt")
file_path_npz = os.path.join(save_path, file_name + f"_mvel{coeff}.npz")


# Plot the trajectory of each motor if needed
if PLOT:
    plt.figure()
    for i in range(motor_goals.shape[1]):
        plt.plot(motor_goals[:, i], label=f"Motor {i+1}")
    plt.title("Motor Goals Trajectory")
    plt.xlabel("Time Step")
    plt.ylabel("Motor Position")
    plt.legend()
    plt.show()

# now set desired initial positions for the motors
# (this value could be set as the Homing position in the Non volatile memory of the Dynamixels)
desired_init_motor_pos = motor_goals[0, :].astype(int)
spiraling_operation.reset_dxl = desired_init_motor_pos
spiraling_operation.reset_dxl_positions()
print("Robot class ready...")
read_motor = spiraling_operation.read_dxl_position()
print("Initial motor positions after reset: ", read_motor)
# Set min and max position for each motor
min_array = np.asarray([40, 100, 20], dtype=int)
max_array = np.asarray([8000, 8000, 8000], dtype=int)

# Actuation_frequency
actuation_sampling_time = 1 / desired_freq[0]

goal_position = np.asarray([0, 0, 0], dtype=int)
goal_position = desired_init_motor_pos

init_motor_position = spiraling_operation.read_dxl_position()
print("Initial motor positions: ", init_motor_position)
print("Goal positions: ", goal_position)

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
    total_time_cycle = []

    goals_tosave = []
    pos_tosave = []
    quar_tosave = []
    motor_pos_tosave = []
    sample_time_idx = 0
    try:
        with open(file_path, "w") as f:
            for itr_n in range(0, num_samples):

                time_total0 = time.time()

                # ------- actuate the muscle
                time_send_cmd0 = time.time()
                goal_position[INDECES_MOTOR_TO_ACTUATE] = motor_goals[
                    itr_n, INDECES_MOTOR_TO_ACTUATE
                ].astype(int)
                # Make sure the goal positions are within the limits
                goal_position = np.clip(goal_position, min_array, max_array)
                spiraling_operation.write_dxl_goal_velocity(
                    vel=custom_velocity.astype(int), pos=goal_position.astype(int)
                )
                time_send_cmd.append(time.time() - time_send_cmd0)

                # Forword model prediction (before any position is received)
                time_receive_obs_and_save0 = time.time()

                # Read all rigid bodies (populated by a call back)
                pos_cur = positions
                quat_curr = quaternions

                # Parse the optitrack readings (assuming 5 rigid bodies are sensed)
                p0 = pos_cur[1]
                p1 = pos_cur[2]
                p2 = pos_cur[3]
                p3 = pos_cur[4]
                p4 = pos_cur[5]
                p5 = pos_cur[6]
                p6 = pos_cur[7]
                p7 = pos_cur[8]

                pos = np.vstack([p0, p1, p2, p3, p4, p5, p6, p7])
                print(f"pos= {pos}")

                # Save the sample index, the commanded motor goals and the
                # measured marker positions
                f.write(str([sample_time_idx, motor_goals[itr_n, :], pos]).strip("[]"))
                f.write("\n")
                sample_time_idx += 1

                q0 = quat_curr[1]
                q1 = quat_curr[2]
                q2 = quat_curr[3]
                q3 = quat_curr[4]
                q4 = quat_curr[5]
                q5 = quat_curr[6]
                q6 = quat_curr[7]
                q7 = quat_curr[8]

                # Update lists to be saved in a npz file
                goals_tosave.append(motor_goals[itr_n, :])
                pos_temp_matrix = np.hstack([p0, p1, p2, p3, p4, p5, p6, p7])
                pos_tosave.append(pos_temp_matrix)
                quar_tosave.append(np.hstack([q0, q1, q2, q3, q4, q5, q6, q7]))
                # Save motors encoder position
                read_motor = spiraling_operation.read_dxl_position()
                motor_pos_tosave.append(read_motor)

                time_receive_obs_and_save.append(
                    time.time() - time_receive_obs_and_save0
                )

                # Sleep
                time_to_sleep = (
                    actuation_sampling_time
                    - time_send_cmd[-1]
                    - time_receive_obs_and_save[-1]
                )
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

                total_time_cycle.append(time.time() - time_total0)

    except KeyboardInterrupt:
        # Close file if it was opened
        if not f.closed:
            f.close()

        ## Convert to array for an alternative save format that contains also the orientations
        act_list = np.array(goals_tosave)
        pos_list = np.array(pos_tosave)
        quat_list = np.array(quar_tosave)
        motor_pos_list = np.array(motor_pos_tosave)
        total_time_cycle = np.array(total_time_cycle)

        # Save to file
        np.savez(
            file_path_npz,
            act_array=act_list,
            pos_array=pos_list,
            quat_array=quat_list,
            total_time_cycle=total_time_cycle,
            motor_pos=motor_pos_list,
        )

        # Sleep for 2 seconds before exiting
        time.sleep(2.0)
        # Set motor to zero
        for _ in range(2):
            spiraling_operation.reset_dxl_positions()
            sleep(3.0)
            spiraling_operation.torque_disable()

        print("Finished.")

        exit()
    ## Convert to array for an alternative save format that contains also the orientations
    act_list = np.array(goals_tosave)
    pos_list = np.array(pos_tosave)
    quat_list = np.array(quar_tosave)
    motor_pos_list = np.array(motor_pos_tosave)
    total_time_cycle = np.array(total_time_cycle)

    # Save to file
    np.savez(
        file_path_npz,
        act_array=act_list,
        pos_array=pos_list,
        quat_array=quat_list,
        total_time_cycle=total_time_cycle,
        motor_pos=motor_pos_list,
    )

    # Sleep for 2 seconds before exiting
    time.sleep(2.0)
    # Set motor to zero
    for _ in range(2):
        spiraling_operation.reset_dxl_positions()
        sleep(3.0)
        spiraling_operation.torque_disable()

    print("Finished.")

    exit()
