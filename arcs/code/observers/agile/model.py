import torch
import torch.nn as nn


# input format: x = [pos_rel, vel_rel] (relative state vector)
# output format: y = f_dw_pred xyz-force of downwash


class AgilePredictor(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, output_dim=6):
        """
        Considers all necessary measurements 
        input v0: [dx,dy,dz, v1x,v1y,v1b, v2x,v2y,v2z, T1x,T1y,T1z, T2x,T2y,T2z] (R15)
        input v1: [dx,dy,dz, v1_norm,v2_norm, v1v2_alpha, T1x,T1y,T1z, T2x,T2y,T2z] (R12)
        output: [fz,torque_roll, torque_pitch, torque_yaw]
        """
        super(AgilePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                         
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),                         
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),                         
            nn.Linear(hidden_dim, output_dim)  
        )

    def forward(self, x):
        return self.model(x)


    

    def evaluate(self, rel_state_vectors):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (dw_x, dw_y, dw_z) 
        """
        with torch.no_grad():
            inputs = torch.tensor(rel_state_vectors).to(torch.float32)
            dw_forces = self.forward(inputs)
            return dw_forces.detach().cpu().numpy()
        


class AgileLeanPredictor(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=3):
        """
        just learn with (dx, abs(v1), abs(v2), T1, T2) (R^11)

        for geometric equivariance: 
        input is
        |proj_xy(T1)|, |proj_xy(T2)|, proj_z(T1), proj_z(T2), angle(proj_xy(T1),proj_xy(T2)), 
        |proj_xy(dp)|, angle(proj_xy(T1), proj_xy(dp))

        after transformation, R^9

        output: [f_z, torque_roll, torque_pitch]
        """
        super(AgileLeanPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                         
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),                         
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),                         
            nn.Linear(hidden_dim, output_dim)  
        )

    def forward(self, x):
        return self.model(x)


    

    def evaluate(self, rel_state_vectors):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (dw_x, dw_y, dw_z) 
        """
        with torch.no_grad():
            inputs = torch.tensor(rel_state_vectors).to(torch.float32)
            dw_forces = self.forward(inputs)
            return dw_forces.detach().cpu().numpy()



def compute_angle_with_direction(v1, v2):
    # computes angles of v1 and v2 relative to the positive x-axis
    angle_v1 = np.arctan2(v1[1], v1[0])
    angle_v2 = np.arctan2(v2[1], v2[0])
    
    # Compute the relative angle
    relative_angle = angle_v2 - angle_v1

    # Normalize to the range [-pi, pi]
    return (relative_angle + np.pi) % (2 * np.pi) - np.pi  # do not take abs


