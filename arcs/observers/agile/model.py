import torch, sys
import torch.nn as nn
import numpy as np
sys.path.append('../../utils/')
from utils import *

# input format: x = [pos_rel, vel_rel] (relative state vector)
# output format: y = f_dw_pred xyz-force of downwash
        

class AgileShallowPredictor(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, output_dim=1):
        """
        for geometric equivariance input is
        |proj_xy(T1)|, |proj_xy(T2)|, proj_z(T1), proj_z(T2), angle(proj_xy(T1),proj_xy(T2)), 
        |proj_xy(dp)|, angle(proj_xy(T1), proj_xy(dp))

        after transformation, R^9

        output: [f_z]
        """
        super(AgileShallowPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                         
            nn.Linear(hidden_dim, output_dim), 
        )

    def forward(self, x):
        return self.model(x)


    def evaluate(self, uav_1_state, uav_2_state):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (dw_x, dw_y, dw_z) 
        """

        with torch.no_grad():
            eqv_agile_transform, flipped_idx = equivariant_agile_transform(uav_1_state, uav_2_state, inference=True)
            x_dw_ct = continous_transform(eqv_agile_transform)
            x_ct = x_dw_ct[:,:14]
            
            inputs = torch.tensor(x_ct).to(torch.float32)
            dw_forces = self.forward(inputs).cpu().numpy()
            
            dw_xy_rel = dw_forces[:,:2]
            
            dp_xy = uav_1_state[:,:2] - uav_2_state[:,:2]
            if len(dp_xy) == 1:
                dw_xy = orthogonal_projection(np.array([dp_xy]).reshape(1,-1), np.array([dw_xy_rel]).reshape(1,-1))
                
            else:
                dw_xy = orthogonal_projection(dp_xy, dw_xy_rel)

            # flip y axis of disturbance if true disturbance has been on left before equiv. transform
            #dw_xy[flipped_idx,1] *= -1.0
            dw_z = dw_forces[:,2]


            return np.column_stack((dw_xy[:,0], dw_xy[:,1], dw_z))
        
    
class AgilePredictor(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, output_dim=1):
        """
        for geometric equivariance input is
        |proj_xy(T1)|, |proj_xy(T2)|, proj_z(T1), proj_z(T2), angle(proj_xy(T1),proj_xy(T2)), 
        |proj_xy(dp)|, angle(proj_xy(T1), proj_xy(dp))

        after transformation, R^9

        output: [f_z]
        """
        super(AgilePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                         
            nn.Linear(hidden_dim, 64), 
            nn.ReLU(),                         
            nn.Linear(64, hidden_dim),
            nn.ReLU(),                         
            nn.Linear(hidden_dim, output_dim)  
        )

    def forward(self, x):
        return self.model(x)


    def evaluate(self, uav_1_state, uav_2_state):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (dw_x, dw_y, dw_z) 
        """

        with torch.no_grad():
            eqv_agile_transform, flipped_idx = equivariant_agile_transform(uav_1_state, uav_2_state, inference=True)
            x_dw_ct = continous_transform(eqv_agile_transform)
            x_ct = x_dw_ct[:,:14]
            
            inputs = torch.tensor(x_ct).to(torch.float32)
            dw_forces = self.forward(inputs).cpu().numpy()
            
            dw_xy_rel = dw_forces[:,:2]
            dp_xy = uav_1_state[:,:2] - uav_2_state[:,:2]
            if len(dp_xy) == 1:
                dw_xy = orthogonal_projection(np.array([dp_xy]).reshape(1,-1), np.array([dw_xy_rel]).reshape(1,-1))
                
            else:
                #print("shape dp", dp_xy.shape)
                dw_xy = orthogonal_projection(dp_xy, dw_xy_rel)

            # flip y axis of disturbance if true disturbance has been on left before equiv. transform
            #dw_xy[flipped_idx,1] *= -1.0
            dw_z = dw_forces[:,2]


            return np.column_stack((dw_xy[:,0], dw_xy[:,1], dw_z))


class AgileContinousPredictor(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, output_dim=1):
        """
        just learn with (dx, abs(v1), abs(v2), T1, T2) (R^11)

        for geometric equivariance: 
        input is
        |proj_xy(T1)|, |proj_xy(T2)|, proj_z(T1), proj_z(T2), angle(proj_xy(T1),proj_xy(T2)), 
        |proj_xy(dp)|, angle(proj_xy(T1), proj_xy(dp))

        after transformation, R^9

        output: [f_z, torque_roll, torque_pitch]
        """
        super(AgileContinousPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                         
            nn.Linear(hidden_dim, 64), 
            nn.ReLU(),                         
            nn.Linear(64, hidden_dim),
            nn.ReLU(),                         
            nn.Linear(hidden_dim, output_dim)  
        )

    def forward(self, x):
        return self.model(x)


    def evaluate(self, uav_1_state, uav_2_state):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (dw_x, dw_y, dw_z) 
        """

        with torch.no_grad():
            equiv_transform = continous_transform(equivariant_agile_transform(uav_1_state, uav_2_state))
            inputs = torch.tensor(equiv_transform).to(torch.float32)
            dw_forces = self.forward(inputs)
            force_list = dw_forces.detach().cpu().numpy()

            return np.column_stack((np.zeros((len(force_list), 2)), force_list))