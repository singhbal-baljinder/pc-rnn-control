import numpy as np
# Create 4x4 transformation matrix from quaternion and position
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_it(input_data, total_p_min=None, total_p_max=None, scaling_factor=1, use_percentage_change_limit=False, min_percentage_change=1):

    if input_data is None or total_p_max is None:
        total_p_min = np.zeros(input_data.shape[1])
        total_p_max = np.zeros(input_data.shape[1])
        for i in range(input_data.shape[1]):
            total_p_min[i] = round(min(input_data[:, i]), 5)
            total_p_max[i] = round(max(input_data[:, i]), 5)
        total_p_min, total_p_max = np.array(total_p_min), np.array(total_p_max)
    
    # If the total min and max change is below a minimum percentage change, set the min and max to the same value
    # This allows to reject some small measurement errors

    if use_percentage_change_limit:
        for i in range(input_data.shape[1]): 
            max_abs =  np.max([np.abs(total_p_max[i]), np.abs(total_p_min[i])])
            if max_abs !=0:
                delta_percentage = ((total_p_max[i] - total_p_min[i])/max_abs)*100
                if delta_percentage < min_percentage_change:
                     total_p_max[i] = total_p_min[i]

    if input_data.ndim == 1:
        input_array = np.zeros((1, len(input_data)))
        for ind in range(len(input_data)):
            input_array[0, ind] = ((1 / (total_p_max[ind] - total_p_min[ind])) * (input_data[ind] - total_p_min[ind]))
            input_array[0, ind] = input_array[0, ind] * scaling_factor

    elif input_data.ndim == 2:
        input_array = np.zeros((input_data.shape[0], input_data.shape[1]))
        for ind in range(input_data.shape[1]):
            # If the min and max are different
            if total_p_max[ind] != total_p_min[ind]:
                input_array[:, ind] = ((1 / (total_p_max[ind] - total_p_min[ind])) * (input_data[:, ind] - total_p_min[ind]))
                input_array[:, ind] = input_array[:, ind] * scaling_factor
            else:
                # If the min and max are the same, set the normalized value to the min = 0
                input_array[:, ind] = 0

    else:
        raise ValueError(f"Array must be 1D or 2D, but got {input_data.ndim}D instead.")

    return input_array, total_p_min, total_p_max


def un_normalize_it(input_data, total_p_min, total_p_max, scaling_factor=1):
    if input_data.ndim == 1:
        input_array = np.zeros((1, len(input_data)))
        for ind in range(len(input_data)):
            input_data[ind] = input_data[ind] / scaling_factor
            input_array[0, ind] = total_p_min[ind] + ((input_data[ind]) * (total_p_max[ind] - total_p_min[ind]))

    elif input_data.ndim == 2:
        input_array = np.zeros((input_data.shape[0], input_data.shape[1]))
        for ind in range(input_data.shape[1]):
            input_data[:, ind] = input_data[:, ind] / scaling_factor
            input_array[:, ind] = total_p_min[ind] + ((input_data[:, ind]) * (total_p_max[ind] - total_p_min[ind]))

    else:
        raise ValueError(f"Array must be 1D or 2D, but got {input_data.ndim}D instead.")

    return input_array

def create_reference_frame(A_p_B, A_q_B):
    '''
    Given position and quaternion, return the 4x4 transformation matrix
    
    :param A_p_B: 3x1 translational position vector
    :param A_q_B: 4x1 quaternion vector   
    '''

    rotation_matrix = Rotation.from_quat(A_q_B).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = A_p_B
    
    return T
    
def change_reference_frame_dataset(G_p_i, G_q_i, G_T_A):
    '''
    Given a trajectory of multiple frames i wrt frame G
    and a transformation matrix G_T_A, compute the trajectory wrt frame A.
    
    :param G_p_i: (num_samples, num_frames x 3) positions of frames i wrt G
    :param G_q_i: (num_samples, num_frames x 4) orientation (quaternion) of frames i wrt G
    :param G_T_A: (4, 4) transformation matrix from frame A to G
    '''
    A_T_G = np.linalg.inv(G_T_A)
    num_samples = G_p_i.shape[0]
    num_frames = G_p_i.shape[1] // 3
    
    A_p_i = np.zeros_like(G_p_i)
    A_q_i = np.zeros_like(G_q_i)
    
    for sample in range(num_samples):
        for frame in range(num_frames):
            idx = frame * 3
            G_T_i = create_reference_frame(G_p_i[sample, idx:idx+3],
                                           G_q_i[sample, frame*4:(frame+1)*4])
            A_T_i = A_T_G @ G_T_i
            A_p_i[sample, idx:idx+3] = A_T_i[:3, 3]
            A_q_i[sample, frame*4:(frame+1)*4] = Rotation.from_matrix(A_T_i[:3, :3]).as_quat()
    
    return A_p_i, A_q_i

