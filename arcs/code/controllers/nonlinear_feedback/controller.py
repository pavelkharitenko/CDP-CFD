import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.append('../../utils/')
from utils import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NonlinearFeedbackController:
    """Sets UAV via attitude controller to fly with certain force in each x,y,z-axis"""
    def __init__(self, uav_mass=3.035, target_yaw=0.0, dt=0.005):
        self.dt = dt
        self.uav_mass = uav_mass

        self.target_yaw = target_yaw
        self.g = 9.81
        self.throttle_hover = 0.3347 # for 80.0, and 0.05 res
        self.TWR = self.throttle_hover / (self.uav_mass * self.g) # thrust to weight ratio

        self.s_int = 0.0

        #self.M = np.eye(3) * self.uav_mass
        self.G = self.uav_mass * np.array([0, 0, self.g])

        self.Kp = 0.5
        self.Ki = 0.0
        self.Lambda = 3.0

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)


    
    def feedback(self, pos, vel, acc):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        

    def position_controller(self, desired):
        """
        Reads current error from reference position and returns target x,y,z forces.
        From: "Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds" O'Connell et al.
        u = Mq *q_dot_dot + g  - K*s - K_i*Int(s)

        desired: R^9 vector of p_des, v_des, a_des
        """
        x_err = self.pos - desired[:3] # p_tilde
        v_err = self.vel - desired[3:6] # p_dot_tilde
        s = v_err - self.Lambda * x_err
        
        self.s_int = s * self.dt + self.s_int
        
        #yaw_err = self.target_yaw - ref[6]

        acc_ref = desired[6:9] - self.Lambda * v_err # see NeuralLander for this term
        
        self.G = 0
        u = self.uav_mass * acc_ref + self.G - self.Kp * s - self.Ki * self.s_int
        f_xyz = u

        return f_xyz


    def set_xyz_force(self, x_force, y_force, z_force):
        """ Setting 0,0,0 means hovering"""

        x_acc = x_force/self.uav_mass + 1e-12
        y_acc = y_force/self.uav_mass + 1e-12
        z_acc = z_force/self.uav_mass + 1e-12

        pitch = np.arctan((x_acc*np.cos(self.target_yaw) + y_acc * np.sin(self.target_yaw))
                         / (z_acc + self.g))
        
        
        roll = np.arctan((np.sin(pitch)*(y_acc*np.cos(self.target_yaw) - x_acc*np.sin(self.target_yaw)))
                         / (x_acc * np.cos(self.target_yaw) + y_acc * np.sin(self.target_yaw)))

        thrust = self.uav_mass*np.sqrt(x_acc**2 + y_acc**2 + (z_acc + self.g)**2) # magnitude
        
        return roll, pitch, self.target_yaw, thrust
    
    def nonlinear_feedback(self, desired):
        f_xyz = self.position_controller(desired)

        print("F_xyz from position controller:", f_xyz)

        
        return self.set_xyz_force(*f_xyz)
        #return self.set_xyz_force(.0,.0,.0)

        

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

        
        

#fc = ForceController()

#fc.set_xyz_force(0,0,-10)


