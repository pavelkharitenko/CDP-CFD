#----------------------------------
# Empirical DW model implemented based on Jain et al. 
# Modeling of aerodynamic disturbances for proximity flight of multirotors
#----------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
sys.path.append('../../utils/')


from utils import *

class EmpiricalPredictor():
    def __init__(self):
        # params for dw model due to drag
        self.Aq = 0.0552503 # uav surface area, square meters
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
            body_frame_vertices, arm_ur_vertices, arm_ll_vertices, arm_ul_vertices, arm_lr_vertices])
        #plt.scatter(self.grid_points[:, 0], self.grid_points[:,1], color="red", s=5)
        #plt.show()

        self.A_cell_horizontal = self.x_cell_length * self.y_cell_length

        self.l = 0.3 # uav arm length
        self.L = 2.0*self.l # vehicle size
        self.c_ax = 5.0 # axial constant
        self.c_rad = 7.3
        self.z0 = 0.22 # virtual origin
        #self.z0 = 0.22
        # with z0 = L and rel_z <0.3, DW force is around 0 to -1N

        self.T = 34.0/4.0

        #print("Horizontal area", self.A_cell_horizontal*len(self.grid_points))


    def velocity_center(self, z):
        induced_vel_propeller = np.sqrt(self.T / (2.0*self.rho*self.Ap))
        Vmax = induced_vel_propeller * self.c_ax*self.L / (z - self.z0)

        return Vmax

    def velocity_field(self, z,r):
        if z == 0.0:
            return 0
        vel = self.velocity_center(z)*np.exp(-self.c_rad*np.square(r/z - self.z0))
        return vel
    
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
        norm = np.linalg.norm(normal_vector)
        if norm == 0.0:
            normal_vector = [0.0,0.0,1.0]
        else:
            normal_vector = normal_vector / norm
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
            # vertical separation from uav 1
            z_dash = u1_z - u2_z

            # compute force f acting on one gridpoint by integrating velocity field at this cell
            f_cell = A_cell_xy_projected * 0.5 * self.Cd * self.rho * self.velocity_field(z_dash, r_dash)**2
            f_cell = np.array([0,0,-f_cell])
            Force_total += f_cell

            # calculate torque resulting from cell as the cross product of f_cell and torque arm
            torque = np.cross(torque_arm, f_cell)
            Torque_total += np.array(torque)

            if plot:
                #ax.quiver(x_b2, y_b2, z_b2,0, 0, -f_cell*20, color='r')
                ax.quiver(x, y, z, 0, 0, f_cell, color='r')


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
    
   
    def propeller_thrust_change(self, prop_speed, oncoming_vel):
        
        
        bv = 3.7 * 10e-7 # thrust_delay_coefficient
        return -bv * prop_speed * oncoming_vel**2
    
    def propeller_thrust_change_torque(self):
        pass


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

#ep = EmpiricalPredictor()
#rel_state = [[0,0,1.0,0,-2,0]]
##print(ep.evaluate(rel_state))



#plot_zy_xy_slices_empirical(ep)


# exp_path = r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\data\two-P600-both-moving-100Hz\2024-10-08-15-38-29-Dataset-NDP-2-P600-flush-frank-720.0sec-72001-ts.p"
# exp = load_forces_from_dataset(exp_path)
# uav_1, uav_2 = exp['uav_list']

# ep = EmpiricalPredictor()

# state = 653
# #state = 20

# ep(uav_1.states[state], uav_2.states[state])


