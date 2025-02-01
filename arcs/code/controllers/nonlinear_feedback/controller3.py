import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.append('../../utils/')
from utils import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NonlinearFeedbackController3:
    """Sets UAV via attitude controller to fly with certain force in each x,y,z-axis"""
    def __init__(self, uav_mass=3.035, target_yaw=0.0, dt=0.001):
        self.dt = dt
        self.uav_mass = uav_mass

        self.target_yaw = target_yaw # must be in radians
        self.g = 9.81
        self.throttle_hover = 0.3347 # for 80.0, and 0.05 res
        #self.throttle_hover = 0.3347 + 0.003 # for 80.0, and 0.05 res

        self.TWR = self.throttle_hover / (self.uav_mass * self.g)  # thrust to weight ratio

        self.s2_int = np.zeros(3)
        self.s_old = np.zeros(3)
        self.s_dot = np.zeros(3)


        #self.M = np.eye(3) * self.uav_mass
        self.G = self.uav_mass * np.array([0, 0, self.g])

        self.k_p = np.array([4.0,4.0, 32.0]) #np.array([3.0,3.0, 18.0]) 
        self.k_v = np.array([1.0,1.0, 27.0]) #np.array([3.0,3.0, 15.0]) 

        
        
        
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)

        self.fxyz = np.zeros(3)


    
    def feedback(self, pos, vel, acc):

        self.pos = np.array(pos)
        print("current z pos", pos[2])

        self.vel = np.array(vel)
        self.acc = np.array(acc)
        

    def pc_nf(self, desired, f_ext=np.zeros(3)):
        """
        Nonlin. Feedback from Li et al. "Hybrid model-based and data-driven 
        disturbance prediction for precise quadrotor trajectory tracking"
        """
        
        p_hat = desired[:3] - self.pos
        v_hat = desired[3:6] - self.vel
        a_fb = self.k_p * p_hat + self.k_v * v_hat
        a_r = desired[6:9]
        m = self.uav_mass 
        

        f_des = m*a_fb + m*a_r + self.G - f_ext

        # advanced feedforward:
        #f_des = m*a_fb + m*a_r + self.G
        #f_des_z = f_ext[2]* -1.0
        #print("F_res", f_des_z)
        #print("F_des", f_des[2])

        #f_des = np.array([f_des[0], f_des[1], f_des[2] + np.clip(f_des_z - f_des[2], 0, f_des_z)])
        #print("F_des + Fres", f_des[2])

        
        return f_des


    def set_xyz_force(self, x_force, y_force, z_force):
        """ 
        Converts f_xyz vector to required roll, pitch, yaw and thrust for attitude controllers.
        Implemented acc. to Filho et al. "Trajectory Tracking for a Quadrotor System: A Flatness-based 
        Nonlinear Predictive Control Approach".

        Setting 0,0,0 means hovering.
        """
        
        
        x_acc = x_force/self.uav_mass + 1e-12
        y_acc = y_force/self.uav_mass + 1e-12
        z_acc = z_force/self.uav_mass + 1e-12


        pitch = np.arctan((x_acc*np.cos(self.target_yaw) + y_acc * np.sin(self.target_yaw))
                         / (z_acc))#+ self.g))
        
        
        roll = np.arctan((np.sin(pitch)*(y_acc*np.cos(self.target_yaw) - x_acc*np.sin(self.target_yaw)))
                         / (x_acc * np.cos(self.target_yaw) + y_acc * np.sin(self.target_yaw)))

        
        thrust = self.uav_mass*np.sqrt(x_acc**2.0 + y_acc**2.0 + (z_acc)**2.0
                                                                  #+ self.g
                                                                  ) # magnitude

        
        
        #thrust = np.clip(thrust, 0.0, 40.0)
        return -roll, pitch, self.target_yaw, thrust
    


    def nonlinear_feedback(self, desired, feedforward=np.zeros(3), uav_z_force=0):
        f_xyz = self.pc_nf(desired, feedforward)
        self.fxyz = f_xyz
        


        #f_xyz = f_xyz - feedforward
        print("f_xyz with feedforward", f_xyz)
        return self.set_xyz_force(*f_xyz)
        

    def nonlinear_feedback_qt_px4(self, desired, feedforward=np.zeros(3), uav_z_force=0):
        """
        Nonlinear feedback output force, and converted to RPY-thrust for px4 attitude controller
        """
        roll, pitch, yaw, thrust = self.nonlinear_feedback(desired, feedforward, uav_z_force)
        print("thrust:", thrust)

        roll, pitch, yaw = np.array([roll, pitch, yaw]) * 180.0/np.pi # convert to radians

        qw, qx, qy, qz = euler_angles_to_quaternion(roll, pitch, yaw)
        throttle = thrust * self.TWR
        px4_input = (1.0, qw, qx, qy, qz, 0.0, throttle) # This is the actuator input vector

        return px4_input

    def align_vector(self, target_vector, thrust_vector):
        """
        Returns rotation needed to align thrust of uav to target
        """
        rotation, _ = Rotation.align_vectors(target_vector, thrust_vector)

        return rotation
    

    def plot_vectors(self, vectors, colors=None, labels=None):
        """
        Plot 3D vectors using a quiver plot.
        
        Parameters:
        - vectors: List of vectors to plot, each represented as an array-like of shape (3,)
        - colors: Optional list of colors for each vector
        - labels: Optional list of labels for each vector
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up the origin
        origin = np.array([[0, 0, 0]])
        
        # Plot each vector
        for i, vector in enumerate(vectors):
            color = colors[i] if colors else 'b'
            label = labels[i] if labels else None
            ax.quiver(*origin[0], *vector, color=color, label=label)
        
        # Set plot limits for better visualization
        all_points = np.vstack(vectors)
        max_limit = np.max(np.abs(all_points)) * 1.1
        ax.set_xlim([-max_limit, max_limit])
        ax.set_ylim([-max_limit, max_limit])
        ax.set_zlim([-max_limit, max_limit])
        
        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Add legend if labels are provided
        if labels:
            ax.legend()
        
        plt.show()

        
        

# nfc = NonlinearFeedbackController()

# nfc.pos = np.array([0.0,0.0,1.0])

# #roll, pitch, target_yaw, thrust = nfc.set_xyz_force(0.0,29.77,3.0)

# #rpy_angles  = np.array([roll, pitch, target_yaw]) * 180.0/np.pi
# #print("Angles of roll pitch yaw:", np.round(rpy_angles,2), ", thrust:", np.round(thrust,2))

# desired_pva = np.array([1.0,0.0,1.0, 0.0,0.0,0.0, 0.0,0.0,0.0])
# f_xyz = nfc.pc_nf(desired_pva)
# print("f_xyz", f_xyz)

# r,p,y, thrust = nfc.set_xyz_force(*f_xyz)

# print("Thrust for attitude controller", thrust)

