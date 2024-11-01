import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.append('../../utils/')
from utils import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PID_Force_Controller:
    """Sets UAV via attitude controller to fly with certain force in each x,y,z-axis"""
    def __init__(self, uav_mass=3.035):
        self.uav_mass = uav_mass
        self.throttle_hover = 0.364 # res 0.05, max.ref.vel. 80

        

    def set_xyz_force_attitude_px4(self, x_force, y_force, z_force):
        """ Setting 0,0,0 means hovering"""

        z_force += self.uav_mass * 9.81 
        target_vector = (x_force, y_force, z_force)
        target_thrust = np.linalg.norm(target_vector)

        # account biases
        target_thrust -= self.hoverbias
        throttle = thrust_to_throttle_p005_mrv80_close_range(target_thrust)
        rotation = self.align_vector(target_vector, (0,0,target_thrust))


        qx, qy, qz, qw = rotation.as_quat()
        return (1.0, qw, qx, qy, qz, 0.0, throttle)

    def align_vector(self, target_vector, thrust_vector):
        """Returns rotation needed to align thrust of uav to target"""
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

        
    



