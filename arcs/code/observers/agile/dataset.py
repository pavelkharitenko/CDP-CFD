import numpy as np
import torch, sys
from scipy.spatial.transform import Rotation as R
sys.path.append('../../utils/')
from utils import *

class AgileLeanDataset(torch.utils.data.Dataset):
    def __init__(self, experiment_paths):
        self.exp_paths = experiment_paths
        self.extract_ndp_labels()

    def __len__(self):
        return self.N
    
    def __getitem__(self,index):
        return self.x[index,:], self.y[index,:]
        
    def extract_ndp_labels(self):
        x = np.empty((0,10))
        y = np.array([])
        for exp_path in self.exp_paths:
            self.N = 0
            
            
            data = np.load(exp_path)
            uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
            #print(self.equivariant_lean_agile_transform(uav_1_states, uav_2_states).shape)
            x_i = equivariant_lean_agile_transform(uav_1_states, uav_2_states)
            y_i = data['dw_forces'][:,2]
            x = np.vstack((x, x_i))
            y = np.append(y,y_i)

        print("extracted total X data of length:", len(x))
        print("extracted total Y labels of length:", len(y))

        self.x =  torch.tensor(x).to(torch.float32)
        self.y =  torch.tensor(y).to(torch.float32)
        self.N = len(self.x)


class AgileEquivariantDataset(torch.utils.data.Dataset):
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
        for exp_path in self.exp_paths:
            self.N = 0
            
            
            data = np.load(exp_path)
            uav_1_states, uav_2_states = data['uav_1_states'], data['uav_2_states']
            #print(self.equivariant_lean_agile_transform(uav_1_states, uav_2_states).shape)
            x_i = equivariant_agile_transform(uav_1_states, uav_2_states)
            y_i = data['dw_forces'][:,2]
            x = np.vstack((x, x_i))
            y = np.append(y,y_i)

        print("extracted total X data of length:", len(x))
        print("extracted total Y labels of length:", len(y))

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


def compute_thrust_vector(uav_states):
        """
        Returns list of Thrust vectors [x,y,z] rotated accordingly to uav's orientation
        """
        thrusts = rps_to_thrust_p005_mrv80(np.mean(uav_states[:,22:26], axis=1, keepdims=True))
        print(thrusts[:5])
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



