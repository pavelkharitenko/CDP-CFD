sys.path.append('../../utils/')
from utils import *
import numpy as np

class ForceController:
    """Sets UAV via attitude controller to fly with certain force in each x,y,z-axis"""
    def __init__(self, uav_mass=3.035):
        self.uav_mass = uav_mass
        

    def set_xyz_force(self, x_force, y_force, z_force):
        """ Setting 0,0,0 means hovering"""

        z_force += self.uav_mass * 9.81
        target_vector = (x_force, y_force, z_force)
        target_thrust = np.linalg.norm(target_vector)
        
        
        throttle = thrust_to_throttle_p005_mrv80_close_range(target_thrust)
        qx, qy, qz, qw = self.vector_to_quaternion((0,0,1), target_vector)
        return (1.0, qw, qx, qy, qz, 0.0, throttle)

    def vector_to_quaternion(self, initial, target):
        # Normalize the input vectors
        initial = initial / np.linalg.norm(initial)
        target = target / np.linalg.norm(target)

        # Calculate the rotation axis using the cross product
        axis = np.cross(initial, target)
        axis_magnitude = np.linalg.norm(axis)
        
        # Handle the case where the vectors are parallel or anti-parallel
        if axis_magnitude < 1e-8:  # If axis magnitude is very small
            # If initial and target are the same direction, no rotation is needed
            if np.allclose(initial, target):
                return np.array([0, 0, 0, 1])  # Identity quaternion
            else:
                # 180-degree rotation (special case)
                axis = np.array([1, 0, 0]) if abs(initial[2]) < 1e-8 else np.array([0, 1, 0])
                return np.hstack([axis, 0])  # Pure rotation quaternion for 180 degrees

        # Calculate the angle between the vectors
        angle = np.arccos(np.clip(np.dot(initial, target), -1.0, 1.0))

        # Normalize the rotation axis
        axis = axis / axis_magnitude

        # Convert axis-angle to quaternion
        qw = np.cos(angle / 2)
        qx, qy, qz = axis * np.sin(angle / 2)

        return qx, qy, qz, qw

# Example usage
initial_vector = np.array([0, 0, 1])  # For (0, 0, k) assume k = 1
target_vector = np.array([1, 2, 3])   # Replace with your desired (a, b, c)
quaternion = vector_to_quaternion(initial_vector, target_vector)
