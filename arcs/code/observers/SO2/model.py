#----------------------------------
# DW predictor based on Smith et al. SO(2)-Equivariant Downwash Models for Close Proximity Flight
# 
#----------------------------------
import torch, sys
import torch.nn as nn
import numpy as np
from skspatial.objects import Plane, Vector
from skspatial.plotting import plot_3d
import matplotlib.pyplot as plt
sys.path.append('../../utils/')
from utils import *

class ShallowEquivariantPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6,32),
            nn.ReLU(),
            nn.Linear(32,2),
        )

    def forward(self,x):
        return self.layers(x)
    

    def evaluate(self, u1_states, u2_states):
        """
        """

        delta_p_list = u1_states[:,:3] - u2_states[:,:3] 
        v_A_list = u1_states[:,3:6]
        v_B_list = u2_states[:,3:6]

        h_features = h_mapping(delta_p_list, v_B_list, v_A_list)
        with torch.no_grad():

            h_inputs = torch.tensor(h_features).to(torch.float32)
            dp_tensor = torch.tensor(delta_p_list).to(torch.float32)

            predicted = self.forward(h_inputs)
            dw_forces = F_tnsr(dp_tensor, predicted).detach().cpu().numpy()
            
            return dw_forces


# helper methods

def proj_A(vec):
    vector = Vector(vec)
    plane = Plane([0, 0, 0], [0, 0, 1]) # A's x, y for now always xy-plane
    projected_vec = plane.project_vector(vector)
    return projected_vec

def proj_A_z(vec):
    vector = Vector(vec)
    a_z = Vector((0,0,1)) # A_z for now always axis-3 (up)
    projected_vec = a_z.project_vector(vector)
    return projected_vec


def h(rel_pos, v_suff, v_prod):
    """
    maps R^9 -> R^6: invariant representation regarding rot. on z-axis.
    """
    projA_dp = proj_A(rel_pos)
    projA_v_b = proj_A(v_suff)
    projA_v_a = proj_A(v_prod)

    result = [
        projA_dp.dot(projA_v_b) / (projA_dp.norm() * projA_v_b.norm()),
        projA_dp.norm(),
        projA_dp.norm(),
        rel_pos[2], # for now, aRm is identity matrix
        v_suff[2],
        projA_v_a.norm()
    ]

    return result



def F(rel_state, f_result):
    """
    Maps relative uav states to dw force predictions
    """
    dp = proj_A(rel_state[:3])
    phi_x = dp.angle_between([1,0,0]) # for now, A's a1 coord is always x-axis
    result =  [
        f_result[0]*np.cos(phi_x),
        f_result[0]*np.sin(phi_x),
        f_result[1]
    ]

    return result

def F_tnsr(delta_position, f_results):
    """
    Maps relative uav states to dw force predictions, R^2 to R^3
    """

    # compute polar angle of proj_xy(dp) w.r.t. x1-axis
    angles = torch.atan2(delta_position[:, 1], delta_position[:, 0])  
    angles = angles % (2 * torch.pi)  # Normalize to [0, 2π)

    # extract force components
    f1 = f_results[:, 0]  # magnitude in x1-x2 plane
    f2 = f_results[:, 1]  # along x3-axis

    # project forces into 3D space
    x1 = torch.cos(angles) * f1
    x2 = torch.sin(angles) * f1
    x3 = f2

    dw_forces = torch.stack((x1, x2, x3), dim=1)

    return dw_forces