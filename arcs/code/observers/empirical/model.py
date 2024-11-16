#----------------------------------
# Empirical DW model implemented based on Jain et al. 
# Modeling of aerodynamic disturbances for proximity flight of multirotors
#----------------------------------
import sys
import numpy as np
from discretization import discretize_shapes
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
        self.c_rad = 25.0
        #self.z0 = 1*self.L # virtual origin
        self.z0 = 0.22
        # with z0 = L and rel_z <0.3, DW force is around 0 to -1N

        self.T = 34.0/4

        #print("Horizontal area", self.A_cell_horizontal*len(self.grid_points))


    def velocity_center(self, z):
        induced_vel_propeller = np.sqrt(self.T / (2*self.rho*self.Ap))
        Vmax = induced_vel_propeller * self.c_ax*self.L / (z - self.z0)

        return Vmax

    def velocity_field(self, z,r):
        vel = self.velocity_center(z)*np.exp(-self.c_rad*np.square(r/z - self.z0))
        return vel
    
    def F_drag(self, uav_1_state, uav_2_state, plot=True):
        Force_total = np.zeros(3)
        Torque_total = 0

        # extract states and rotations
        u1_x, u1_y, u1_z = uav_1_state[0:3] # uav 1 position
        u2_x, u2_y, u2_z = uav_2_state[0:3] # uav 2 position
        u2_yaw, u2_pitch, u2_roll = uav_2_state[9:12] # uav 2 orientation
        b2Rw = R.from_euler('xyz', [u2_roll, u2_pitch, u2_yaw + self.bias_yaw], degrees=False)

        # for testing, put position of uav 1 above uav2  -----
        u1_x, u1_y, u1_z = uav_2_state[0:3]
        #u1_x += 0.25
        u1_z += 1.0
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

        print("total Drag z-Force on UAV2:",Force_total)
        print("total Drag torque on UAV2:", Torque_total)

        if plot:
            
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
    
    def T_drag(self, uav_1_state, uav_2_state):
        torque_total = np.array([0.0,0.0,0.0])
        F_total = 0

        # extract states and rotations
        u1_x, u1_y, u1_z = uav_1_state[0:3]
        u2_x, u2_y, u2_z = uav_2_state[0:3]
        u2_yaw, u2_pitch, u2_roll = uav_2_state[9:12]

        # for testing, put position of uav 1 above uav2  -----
        u1_x, u1_y, u1_z = uav_2_state[0:3]
        #u1_y += 0.5
        u1_x += 0.25
        u1_z += 0.5
        # ---

        #b2Rw = R.from_euler('xyz', [0 , 0, np.pi/4.0], degrees=False)
        b2Rw = R.from_euler('xyz', [u2_roll, u2_pitch, u2_yaw + self.bias_yaw], degrees=False)

        # update cell surface regarding z-Plane
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
        
        
        # Plotting setup
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d.proj3d import proj_transform
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
                        
        
        for grid_pnt in self.grid_points:
            # rotate gridpoint vector to uav frame
            x, y, z = grid_pnt[0], grid_pnt[1], 0
            x, y, z = b2Rw.apply([x,y,z])
            x,y,z = x + u2_x, y + u2_y, z + u2_z

            # ----
            
            # horizontal separation from uav 1
            r_dash = np.sqrt((u1_x - x)**2 + (u1_y - y) **2)
            
            # vertical separation from uav 1
            z_dash = u1_z - u2_z

            # distance uav 2 center and gridpoint
            r_dash_abs = np.array([x, y, z]) # gp
            r = np.array([u2_x, u2_y, u2_z]) # uav 2 center
            
            # torque arm
            t_arm = r_dash_abs - r

            # vertical separation
            F = np.array([0,0,-1])* A_cell_xy_projected * 0.5 * self.Cd * self.rho * self.velocity_field(z_dash, r_dash)**2

            F_total += F
            #torque_int += torque_cell


            # Define two vectors: `r` (position) and `F` (force)
            torque_arm = t_arm # position vector in 3D


            # Calculate torque as the cross product
            torque = np.cross(torque_arm, F)
            # Define an arbitrary origin for torque application
            torque_origin = r 

            torque_total += np.array(torque)

            normal_vector = torque / np.linalg.norm(torque)

            # Find two orthogonal vectors in the plane of rotation
            # Use Gram-Schmidt to get vectors orthogonal to the normal vector
            v1 = torque_arm / np.linalg.norm(torque_arm)  # normalize position vector
            v2 = np.cross(normal_vector, v1)  # v2 is perpendicular to both normal and v1

            # Define the arc in the plane defined by v1 and v2
            angle = np.linspace(0, -np.pi/2, 100)  # 90-degree arc
            arc_radius = np.linalg.norm(torque_arm)
            arc_points = (arc_radius * np.outer(np.cos(angle), v1) +
                        arc_radius * np.outer(np.sin(angle), v2))

            # Offset the arc points by the torque origin
            arc_points += torque_origin


            # Plot position vector `r` starting from `torque_origin`
            #ax.quiver(*torque_origin, *r, color='blue', label="Position Vector r (from origin)")

            # Plot force vector `F` starting from the end of `r` (torque_origin + r)
            force_origin = torque_origin + torque_arm
            ax.quiver(*force_origin, *F, color='green', label="Force Vector F")

            # Plot torque vector starting from `torque_origin`
            ax.quiver(*torque_origin, *torque, color='red', label="Torque Vector τ = r x F")

            # Plot the arc in the plane as a purple line
            ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='purple', label="Torque Arc")



        # visualize total torque:
        torque_origin = np.array([u2_x, u2_y, u2_z]) 

        normal_vector = torque_total / np.linalg.norm(torque_total)

        # Find two orthogonal vectors in the plane of rotation
        # Use Gram-Schmidt to get vectors orthogonal to the normal vector
        v1 = r / np.linalg.norm(r)  # normalize position vector
        v2 = np.cross(normal_vector, v1)  # v2 is perpendicular to both normal and v1

        # Define the arc in the plane defined by v1 and v2
        angle = np.linspace(0, -np.pi/2, 100)  # 90-degree arc
        arc_radius = np.linalg.norm(torque_total)*1
        #arc_radius = 1.0

        arc_points = (arc_radius * np.outer(np.cos(angle), v1) +
                    arc_radius * np.outer(np.sin(angle), v2))

        # Offset the arc points by the torque origin
        arc_points += torque_origin


        # Plot position vector `r` starting from `torque_origin`
        #ax.quiver(*torque_origin, *r, color='blue', label="Position Vector r (from origin)")

        # Plot force vector `F` starting from the end of `r` (torque_origin + r)
        force_origin = torque_origin + r
        ax.quiver(*force_origin, *F, color='orange', label="Force Vector F")

        # Plot torque vector starting from `torque_origin`
        ax.quiver(*torque_origin, *torque_total, color='red', label="Torque Vector τ = r x F")

        # Plot the arc in the plane as a purple line
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='red', label="Total torque")


        # add uav 1 z-axis
        ax.quiver(u2_x, u2_y, u2_z, 0, 0, -0.2, color='orange', label="UAV 2")

        ax.quiver(u1_x, u1_y, u1_z, 0, 0, -0.2, color='b', label="UAV 1")

        print("Total Torque acting on UAV 2", torque_total)
        print("Total z-Force acting on UAV 2", F_total)



        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-2.5, -1.5])
        ax.set_zlim([2.5, 3.5])
        

        # Labeling
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plt.legend()
        plt.title("Torque and Circular Arrow Visualization")


        plt.show()
            
        
        return torque_total

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


    def evaluate(self, uav_1_states, uav_2_states):
        """
        From R:6 (rel_pos, rel_vel) to R:3 (0, 0, dw_z) 
        """
        predicted_forces = []
        ep = EmpiricalPredictor()

        for states in zip(uav_1_states, uav_2_states):
            predicted_forces.append(ep(states[0], states[1]))


        return predicted_forces


# exp_path = r"C:\Users\admin\Desktop\IDP\CDP-CFD\arcs\code\data_collection\ndp-data-collection\data\two-P600-both-moving-100Hz\2024-10-08-15-38-29-Dataset-NDP-2-P600-flush-frank-720.0sec-72001-ts.p"
# exp = load_forces_from_dataset(exp_path)
# uav_1, uav_2 = exp['uav_list']

# ep = EmpiricalPredictor()

# state = 653
# #state = 20

# ep(uav_1.states[state], uav_2.states[state])


