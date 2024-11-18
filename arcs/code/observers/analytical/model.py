#----------------------------------
# Empirical DW model implemented based on Bauersfeld et al. 
# Modeling of aerodynamic disturbances for proximity flight of multirotors
# and Chee et al. 
# Flying Quadrotors in Tight Formations using Learning-based Model Predictive Control
#----------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.append('../../utils/')


from utils import *

class AnalyticalPredictor():
    def __init__(self):
        
        # empirical parameters
        self.Bd = 10.11
        self.S = 0.07668
        self.s0 = 0.2#-5.817

        self.Cd = 1.18 # aerodynamic property of uav (like flat plane)
        self.Ap = np.pi * 0.195**2 # propeller disk area A = pi * r^2
        self.rho = 1.204 # air density

        self.bias_yaw = np.pi/2.0 # discrepancy cad model and AR orientation

        body_frame_vertices = [(-0.03, 0.1305), (0.03, 0.1305), (0.0865, 0.0735), (0.0865, 0.0575), 
                            (0.07, 0.0375), (0.07, -0.0375),(0.0865,  -0.0575), (0.0865,  -0.0735), 
                     (0.026, -0.1305), (-0.026, -0.1305), (-0.0865,  -0.0735), (-0.0865,  -0.0575),
                            (-0.07, -0.0375),(-0.07, 0.0375), (-0.0865, 0.0575), (-0.0865, 0.0735), ]
        
        arm_ur_vertices = np.array([(0.0865, 0.0735), (0.203173, 0.190173), (0.210244, 0.183102),
                                 (0.238528,0.211386), (0.210244, 0.239670), (0.181960, 0.211386),
                                                      (0.189031,0.204315), (0.0723584,0.0876424), ])
        arm_ll_vertices = -arm_ur_vertices
        arm_ul_vertices = np.array([(-vert[0], vert[1]) for vert in arm_ur_vertices])
        arm_lr_vertices = -arm_ul_vertices
        
        self.grid_points, self.x_cell_length, self.y_cell_length = discretize_shapes([
            body_frame_vertices, arm_ur_vertices, arm_ll_vertices, arm_ul_vertices, arm_lr_vertices],
            n_cells=12.0,)
        #plt.scatter(self.grid_points[:, 0], self.grid_points[:,1], color="red", s=5)
        #plt.show()

        self.A_cell_horizontal = self.x_cell_length * self.y_cell_length

        
        self.l = 2.0 * 0.3 # motor distance L (rotor diagonal distance of a uav)
        

        self.T_hover = 34.0 # mg but in practice mg + bias
        self.U_hover = self.U_h()





    def U_h(self):
        U_h = np.sqrt(self.T_hover / (2.0 * self.rho *  self.Ap * 4.0))
        return U_h
    
    def U_flow(self, s, r):
        # normalize
        s = s / self.l
        r = r / self.l
        #print("r normalized:", r)
        # compute centerline velocity and jet half width spread as
        U_c = self.Bd /(s - self.s0)
        r_half = self.S * (s - self.s0)
        #print("r_half", r_half)
        r_half = r_half / self.l
        #print("r half normalized", r_half)

        r_tilde = r / r_half

        #print("r_tilde", r_tilde)

        # scale Uc
        U_c = U_c * self.U_hover
        #print("U hover", self.U_hover)

        # compute velocity field:
        #print("denominator", (1.0 + (np.sqrt(2.0) - 1.0)* r_tilde**2)**2)
        U_c = U_c / (1.0 + (np.sqrt(2.0) - 1.0)* r_tilde**2)**2

        #print("U_c", U_c)
        return U_c


    
    def F_drag(self, uav_1_state, uav_2_state, plot=False):
        Force_total = np.zeros(3)
        Torque_total = 0

        # extract states and rotations
        u1_x, u1_y, u1_z = uav_1_state[0:3] # uav 1 position
        u2_x, u2_y, u2_z = uav_2_state[0:3] # uav 2 position
        u2_yaw, u2_pitch, u2_roll = uav_2_state[9:12] # uav 2 orientation
        b2Rw = R.from_euler('xyz', [u2_roll, u2_pitch, u2_yaw + self.bias_yaw], degrees=False)

        # for testing, put position of uav 1 above uav2  -----
        #u1_x, u1_y, u1_z = uav_2_state[0:3]
        #u1_x += 0.25
        #u1_z += 1.0
        # ---

        # update cell surface regarding z-Plane:
        x_cell_b = b2Rw.apply([self.x_cell_length, 0,0])
        y_cell_b = b2Rw.apply([0,self.y_cell_length, 0])
        # 1 compute the normal vector of the cell plane using cross product
        normal_vector = np.cross(x_cell_b, y_cell_b)
        # 2 normalize plane's normal vector
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        # 3 compute the cosine of the angle with the Z-axis
        cosine_angle = abs(normal_vector[2])
        A_cell_xy_projected = self.A_cell_horizontal * cosine_angle
        #print("New proj. Acell:", A_cell_xy_projected * len(self.grid_points))


        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # iterate discretized points and compute z-force and torque from velocity field model
        for grid_pnt in self.grid_points:

            # extract, rotate and translate gridpoints of uav 2
            x, y, z = grid_pnt[0], grid_pnt[1], 0
            x, y, z = b2Rw.apply([x,y,z])
            x, y, z = x + u2_x, y + u2_y, z + u2_z

            # torque arm: distance grid point and uav 2 center
            torque_arm = np.array([x, y, z]) - np.array([u2_x, u2_y, u2_z])

            # horizontal separation of grid point from uav 1
            r_dash = np.sqrt((u1_x - x)**2 + (u1_y - y)**2)
            #print(r_dash)
            # vertical separation from uav 1
            z_dash = u1_z - u2_z

            # compute force f acting on one gridpoint by integrating velocity field at this cell
            f_cell = A_cell_xy_projected * 0.5 * self.Cd * self.rho * self.U_flow(z_dash, r_dash)**2
            
            f_cell = np.array([0,0,-f_cell])
            Force_total += f_cell

            # calculate torque resulting from cell as the cross product of f_cell and torque arm
            torque = np.cross(torque_arm, f_cell)
            Torque_total += np.array(torque)

            if plot:
                #ax.quiver(x_b2, y_b2, z_b2,0, 0, -f_cell*20, color='r')
                ax.quiver(x, y, z, 0, 0, -f_cell*4, color='r')


        if plot:
            print("total Drag z-Force on UAV2:",Force_total)
            print("total Drag torque on UAV2:", Torque_total)
            
            ax.quiver(u2_x, u2_y, u2_z, 0, 0, -0.2, color='orange', label="UAV 2")
            ax.quiver(u1_x, u1_y, u1_z, 0, 0, -0.2, color='b', label="UAV 1")
            # set plot limits and labels
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-2.5, -1.5])
            ax.set_zlim([2, 3])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.show()


        return Force_total, Torque_total
    
  
    


    def __call__(self, uav_1_state, uav_2_state):
        """
        Computes resulting DW force on bottom quadrotor
        """
        F_drag_total, T_drag_total = self.F_drag(uav_1_state, uav_2_state)
        #T_drag_total = self.T_drag(uav_1_state, uav_2_state)
        return F_drag_total, T_drag_total


    def evaluate(self, rel_states):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (0, 0, dw_z) 
        Only evaluates F_drag, T_drag unused but possible to account
        """
        predicted_forces = []
        

        for u1_state in rel_states:
            u2_state = np.zeros(12)
            u1_state = np.pad(u1_state, (0, 12 - len(u1_state)), mode='constant', constant_values=0)
            #print(u1_state)
            #print(u2_state)

            predicted_forces.append(self.F_drag(u1_state, u2_state)[0])


        return np.array(predicted_forces)

#ap = AnalyticalPredictor()
#rel_state = [[0.0, 0.0, 0.5, 0, 0, 0]]
#print(ap.evaluate(rel_state))

#plot_zy_xy_slices_empirical(ap)

# exp_path = r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\data\two-P600-both-moving-100Hz\2024-10-08-15-38-29-Dataset-NDP-2-P600-flush-frank-720.0sec-72001-ts.p"
# exp = load_forces_from_dataset(exp_path)
# uav_1, uav_2 = exp['uav_list']

# ep = AnalyticalPredictor()

# state = 653
# #state = 20

# ep(uav_1.states[state], uav_2.states[state])



