import numpy as np
import torch, sys
import torch.nn as nn
sys.path.append('../../utils/')
from utils import *

# --- Step 2: Neural Network Architecture ---
class NNARXModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        input_dim: Number of input features (lags * state_dim + lags * force_dim)
        hidden_dim: Number of hidden units
        """
        super(NNARXModel, self).__init__()
        self.lags = int(input_dim/14)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single output: y(t+1)
        )

    def forward(self, x):
        return self.network(x)
    
    def evaluate(self, uav_1_state, uav_2_state):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (dw_x, dw_y, dw_z) 
        """
        
        with torch.no_grad():
            equiv_transform = continous_transform(equivariant_agile_transform(uav_1_state, uav_2_state))

            states = equiv_transform





            n_samples, n_states = equiv_transform.shape
            
            # Initialize lists to store lagged states and forces
            lagged_states = []
            state_lags = states[0:self.lags]
            for i in range(self.lags):
                lagged_states.append(state_lags.flatten())
            print(len(state_lags.flatten()))

            #print(lagged_states)
            # add padding for first state
            
            # For each time step, create lagged states and forces
            for i in range(self.lags, n_samples):
                # Create lagged states (each feature has its own lag)
                state_lags = states[i - self.lags:i]  # Shape (lags, n_states)
                #print("state_lags", state_lags)
                
                
                # Flatten the lagged states and forces, but keep them separate for each feature
                lagged_states.append(state_lags.flatten())  # Flatten each feature's lagged values

                #print(len(state_lags.flatten()))
                #exit(0)
                
            # Stack the lagged states and forces into input features
            lagged_states = np.array(lagged_states)
            
            # Combine states and forces to create the final input (lags * n_states + lags * n_forces)
            inputs = np.hstack([lagged_states])
            

            
            
            # Convert inputs and outputs to torch tensors
            inputs = torch.tensor(inputs, dtype=torch.float32)

            dw_forces = self.forward(inputs)
            #dw_forces = dw_forces[]
            force_list = dw_forces.detach().cpu().numpy()
            #force_list = force_list[0:-1:self.lags]
            #print(force_list.shape)

            return np.column_stack((np.zeros((len(force_list), 2)), force_list))
        

# helper functions

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
    #print("T1xy", T1_xy)
    T1_z = T1[:,2]
    
    T2 = compute_thrust_vector(u2_states)
    T2_xy = T2[:,:2]
    T2_z = T2[:,2]

    # symmetry assumptions: flip T1 to left side of dp, and flip T2 accordingly to T2
    T1_xy, dp_xy, T2_xy = process_vectors(T1_xy, dp_xy, T2_xy)
    #print("T1xy", T1_xy)

    angle_T1_T2_xy = compute_signed_angles(T1_xy, T2_xy)
    angle_T1_dp_xy = compute_signed_angles(T1_xy, dp_xy)

    result = (
        np.linalg.norm(dp_xy, axis=1), dp_z, # 0 1
        angle_T1_dp_xy, # 2
        np.linalg.norm(T1_xy, axis=1), T1_z, # 3 4
        np.linalg.norm(T2_xy, axis=1), T2_z, # 5 6
        angle_T1_T2_xy, # 7
        angle_v1_dp_xy, # 8
        np.linalg.norm(v1_xy, axis=1), v1_z, # 9 10
        np.linalg.norm(v2_xy, axis=1), v2_z, angle_v1_v2_xy # 11 12 13
    ) 


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
        #print(thrusts[:5])
        thrust_vectors = np.column_stack((np.zeros(len(thrusts)), np.zeros(len(thrusts)), thrusts))
        rotations = R.from_euler('zyx', uav_states[:,9:12])
        # TODO: pitch angle is negative, but rotation is positive (right hand convention)

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







