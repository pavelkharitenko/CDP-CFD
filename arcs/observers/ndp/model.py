#----------------------------------
# NDP model implemented from Li et al. Nonlinear MPC for Quadrotors in Close-Proximity 
# Flight with Neural Network Downwash Prediction, arXiv:2304.07794v2
# Their implementation found at https://github.com/Li-Jinjie/ndp_nmpc_qd
#----------------------------------
import torch
import torch.nn as nn
import numpy as np

# input format: x = [pos_rel, vel_rel] (relative state vector)
# output format: y = f_dw_pred xyz-force of downwash


class DWPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )

    def forward(self,x):
        return self.linear_relu_stack(x)
    

    def evaluate(self, uav1_states, uav2_states):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (dw_x, dw_y, dw_z) 
        """
        rel_state_vectors = np.array(uav1_states)[:,:6] - np.array(uav2_states)[:,:6]
        with torch.no_grad():
            inputs = torch.tensor(rel_state_vectors).to(torch.float32)
            dw_forces = self.forward(inputs)
            return dw_forces.detach().cpu().numpy()



