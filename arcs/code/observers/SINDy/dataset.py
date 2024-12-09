import numpy as np
import torch, sys
from scipy.spatial.transform import Rotation as R
sys.path.append('../../utils/')
from utils import *




class AgileContinousDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths):
        self.exp_paths = experiment_paths
        self.extract_ndp_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ndp_labels(self):
        x = np.empty((0,14))
        y = np.array([])
        x_rel = np.empty((0,6))
        for exp_path in self.exp_paths:
            self.N = 0
            
            
            data = np.load(exp_path)
            uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
            delta_x =  uav_1_states[:,:6] - uav_2_states[:,:6]
            #print(self.equivariant_lean_agile_transform(uav_1_states, uav_2_states).shape)
            x_i = continous_transform(equivariant_agile_transform(uav_1_states, uav_2_states))
            y_i = data['dw_forces'][:,2]
            x = np.vstack((x, x_i))
            x_rel = np.vstack((x_rel,delta_x))
            y = np.append(y,y_i)

        print("extracted total X data of length:", len(x))
        print("extracted total Y labels of length:", len(y))

        self.x_arr = x
        self.y_arr = y
        self.x_rel = x_rel

        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)
        self.N = len(self.x)

def equivariant_agile_transform(u1_states, u2_states):
    """
    Returns equivariant representation w.r.t Z-axis
    """

    dp_xy = u1_states[:,:2] - u2_states[:,:2]
    dp_z = u1_states[:,2] - u2_states[:,2]

    v1_xy = u1_states[:,3:5]
    v1_z = u1_states[:,5]
    v2_xy = u2_states[:,3:5]
    v2_z = u2_states[:,5]

    v1_xy, _, v2_xy = process_vectors(v1_xy, dp_xy, v2_xy)
    angle_v1_dp_xy = compute_signed_angles(v1_xy, dp_xy)
    angle_v1_v2_xy = compute_signed_angles(v1_xy, v2_xy)



    # compute T1 thrust vectors and their orientations and translations 
    T1 = compute_thrust_vector(u1_states)
    T1_xy = T1[:,:2]
    T1_z = T1[:,2]
    
    T2 = compute_thrust_vector(u2_states)
    T2_xy = T2[:,:2]
    T2_z = T2[:,2]

    # symmetry assumptions: flip T1 to left side of dp, and flip T2 accordingly to T2
    T1_xy, dp_xy, T2_xy = process_vectors(T1_xy, dp_xy, T2_xy)

    angle_T1_T2_xy = compute_signed_angles(T1_xy, T2_xy)
    angle_T1_dp_xy = compute_signed_angles(T1_xy, dp_xy)

    result = (np.linalg.norm(dp_xy, axis=1), dp_z, 
                angle_T1_dp_xy, 
                np.linalg.norm(T1_xy, axis=1), T1_z, 
                np.linalg.norm(T2_xy, axis=1), T2_z, angle_T1_T2_xy, 
                angle_v1_dp_xy, 
                np.linalg.norm(v1_xy, axis=1), v1_z, 
                np.linalg.norm(v2_xy, axis=1), v2_z, angle_v1_v2_xy)


    return np.column_stack(result)

def equivariant_lean_agile_transform(u1_states, u2_states):
    """
    Returns equivariant representation w.r.t Z-axis
    """

    dp_xy = u1_states[:,:2] - u2_states[:,:2]
    dp_z = u1_states[:,2] - u2_states[:,2]

    v1_abs = np.linalg.norm(u1_states[:,3:6], axis=1)
    v2_abs = np.linalg.norm(u2_states[:,3:6], axis=1)

    # compute T1 thrust vectors and their orientations and translations 
    T1 = compute_thrust_vector(u1_states)
    T1_xy = T1[:,:2]
    T1_z = T1[:,2]
    
    T2 = compute_thrust_vector(u2_states)
    T2_xy = T2[:,:2]
    T2_z = T2[:,2]

    # symmetry assumptions: flip T1 to left side of dp, and flip T2 accordingly to T2
    T1_xy, dp_xy, T2_xy = process_vectors(T1_xy, dp_xy, T2_xy)

    angle_T1_T2_xy = compute_signed_angles(T1_xy, T2_xy)

    angle_T1_dp_xy = compute_signed_angles(T1_xy, dp_xy)

    result = (np.linalg.norm(dp_xy, axis=1), dp_z, angle_T1_dp_xy, 
                np.linalg.norm(T1_xy, axis=1), T1_z, 
                np.linalg.norm(T2_xy, axis=1), T2_z, 
                angle_T1_T2_xy, v1_abs, v2_abs)


    return np.column_stack(result)



def continous_transform(equivariant_states):
    """
    Input is columns of |dp_xy|, dp_z, angle_T1_dp, |T1_xy|, T1_z, ...
    """
    T1_decomp = angle_demcomposition(equivariant_states[:,2], equivariant_states[:,3])
    T2_decomp = angle_demcomposition(equivariant_states[:,7], equivariant_states[:,5])
    v1_decomp = angle_demcomposition(equivariant_states[:,8], equivariant_states[:,9])
    v2_decomp = angle_demcomposition(equivariant_states[:,13], equivariant_states[:,11])

    result = (
        # z-axis components and dp length
        equivariant_states[:, 0], # |p_xy|
        equivariant_states[:, 1], # p_z
        equivariant_states[:,4], # T1_z
        equivariant_states[:,6], # T2_z
        equivariant_states[:,10], # v1_z
        equivariant_states[:,12], # v2_z
        # new angle independant sinusoidal encodings
        T1_decomp[:,0], T1_decomp[:,1],
        T2_decomp[:,0], T2_decomp[:,1],
        v1_decomp[:,0], v1_decomp[:,1],
        v2_decomp[:,0], v2_decomp[:,1],
    )

    return np.column_stack(result)


def angle_demcomposition(angles, amplitudes):
    result = np.column_stack((amplitudes * np.sin(angles), amplitudes * np.cos(angles)))
    return result




def compute_thrust_vector(uav_states):
        """
        Returns list of Thrust vectors [x,y,z] rotated accordingly to uav's orientation
        """
        thrusts = rps_to_thrust_p005_mrv80(np.mean(uav_states[:,22:26], axis=1, keepdims=True))
        #print(thrusts[:5])
        thrust_vectors = np.column_stack((np.zeros(len(thrusts)), np.zeros(len(thrusts)), thrusts))
        rotations = R.from_euler('zyx', uav_states[:,9:12])
        thrust_list = rotations.apply(thrust_vectors) 

        return thrust_list



def mirror_vectors(v1, v2):
    """Mirror vectors v1 around vectors v2."""
    # Mirror the vectors v1 around v2 using the formula
    dot_product = np.sum(v1 * v2, axis=1)
    norm_v2_squared = np.sum(v2 * v2, axis=1)
    return 2 * (dot_product / norm_v2_squared).reshape(-1, 1) * v2 - v1

def process_vectors(vectors1, vectors2, vectors3):
    """
    If first vector i is to the left of the second vector, mirror it to the right 
    and mirror the vector i of the third list to the opposite side.
    """
    # if T1 is left of dp, mirror T1 to right, and flip T2
    cross_prod = np.cross(vectors1, vectors2)
    mask = cross_prod > 0.0  # means vector1 is to the left of vector2 (positive cross product)
    
    # Mirror T1 and T2 based on mask
    vectors1[mask] = mirror_vectors(vectors1[mask], vectors2[mask])
    vectors3[mask] = mirror_vectors(vectors3[mask], vectors2[mask])

    # where T1 is collinear to dp, bring T2 to right side
    mask_2 = cross_prod == 0.0
    cross_prod32 = np.cross(vectors3, vectors2)
    mask_3 = cross_prod32 > 0.0 # means T2 is to the left of vector2 (positive cross product)
    v1_colin_and_v3_left = mask_2 & mask_3
    vectors3[v1_colin_and_v3_left] = mirror_vectors(vectors3[v1_colin_and_v3_left], 
                                                   vectors2[v1_colin_and_v3_left])

    return vectors1, vectors2, vectors3



def compute_signed_angles(v1_list, v2_list):
    """
    Computes the signed and normalized angles between corresponding vectors in two lists.
    
    Parameters:
        v1_list (np.ndarray): Array of shape (N, 2), list of 2D vectors.
        v2_list (np.ndarray): Array of shape (N, 2), list of 2D vectors.
    
    Returns:
        tuple: Two arrays of shape (N,) - signed angles and normalized angles.
    """
    # Ensure inputs are numpy arrays
    v1_list = np.array(v1_list)
    v2_list = np.array(v2_list)
    
    # Compute the signed angle using atan2
    angles = np.arctan2(v2_list[:, 1], v2_list[:, 0]) - np.arctan2(v1_list[:, 1], v1_list[:, 0])
    
    # Normalize angles to [0, 2π)
    angles_normalized = angles % (2 * np.pi)
    
    return angles_normalized




def split_sequences_by_threshold(arr, arr2, threshold=1.0, time_step=0.005):
    """
    Splits a numpy array into sequences if arr[:, 1] deviates by more than the threshold.
    Generates a timestamp list for each split and reports split times.

    Parameters:
    - arr: numpy array of shape (n_samples, n_features)
    - threshold: float, the deviation threshold for splitting
    - time_step: float, increment step for timestamps

    Returns:
    - split_sequences: list of numpy arrays (each sequence)
    - timestamp_lists: list of numpy arrays of timestamps for each sequence
    """
    split_sequences = []  # To hold the split data
    split_forces = []
    timestamp_lists = []  # To hold the corresponding timestamps
    split_times = []      # To hold the time when a split occurs
    start_idx = 0         # Start index of the current sequence

    # Iterate through the array to find split points
    for i in range(1, len(arr)):
        # Check if the deviation exceeds the threshold
        if abs(arr[i, 0] - arr[i-1, 0]) > threshold:
            # Split the array from start_idx to i
            split_sequences.append(arr[start_idx:i])
            split_forces.append(arr2[start_idx:i])

            
            # Generate timestamps for the current sequence
            timestamps = np.arange(0, (i - start_idx) * time_step, time_step)
            timestamp_lists.append(timestamps)
            
            # Log the time when the split happened
            split_time = i * time_step
            split_times.append(split_time)
            #print(f"Split at time {split_time:.2f}s (index {i})")

            # Update the start index
            start_idx = i

    # Handle the last sequence
    if start_idx < len(arr):
        split_sequences.append(arr[start_idx:])
        split_forces.append(arr2[start_idx:])

        timestamps = np.arange(0, (len(arr) - start_idx) * time_step, time_step)
        timestamp_lists.append(timestamps)

    # Print total splits
    print(f"\nTotal splits: {len(split_sequences) - 1}")

    return split_sequences[20:], split_forces[20:], timestamp_lists[20:], split_times


