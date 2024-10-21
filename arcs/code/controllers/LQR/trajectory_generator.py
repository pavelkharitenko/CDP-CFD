import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Trajectory:
    def __init__(self, linearization_parameters):
        self.linearization_parameters = linearization_parameters

    def compute_control_inputs(self, desired_state, yaw):
        """
        Compute control inputs (torques and thrust) based on the desired state.

        Args:
            desired_state (tuple): The desired (x, y, z) position to track.
            yaw (float): Current yaw angle of the drone.

        Returns:
            tuple: (roll_torque, pitch_torque, yaw_torque, thrust)
        """
        # Extract desired state
        x, y, z = desired_state
        
        # Control gains
        K_roll = self.linearization_parameters['K_roll']
        K_pitch = self.linearization_parameters['K_pitch']
        K_yaw = self.linearization_parameters['K_yaw']
        
        # Calculate torques based on desired state
        roll_torque = K_roll * (x - self.linearization_parameters['x_target'])
        pitch_torque = K_pitch * (y - self.linearization_parameters['y_target'])
        yaw_torque = K_yaw * (yaw - self.linearization_parameters['yaw_target'])
        
        # Set constant thrust (example: thrust to hover for a 1kg drone)
        thrust = 9.81  # Adjust based on your drone's specifications

        return roll_torque, pitch_torque, yaw_torque, thrust

    def generate_horizontal_eight_trajectory(self, num_points, height):
        """
        Generate a horizontal eight trajectory.

        Args:
            num_points (int): Number of points in the trajectory.
            height (float): The fixed height (z) of the trajectory.

        Returns:
            list: A list of tuples, each containing (roll_torque, pitch_torque, yaw_torque, thrust).
        """
        reference_states = []
        t = np.linspace(0, 2 * np.pi, num_points)

        # Generate horizontal eight points
        x = 2 * np.sin(t)  # Amplitude along x-axis
        y = np.sin(2 * t)  # Amplitude along y-axis
        z = np.full(t.shape, height)  # Constant height

        # Calculate yaw based on trajectory
        yaw = np.arctan2(np.gradient(y), np.gradient(x))  # Calculate yaw angles

        for i in range(num_points):
            control_inputs = self.compute_control_inputs((x[i], y[i], z[i]), yaw[i])
            reference_states.append(control_inputs)

        return reference_states, x, y, z, yaw

    def generate_forward_flight_trajectory(self, num_points, velocity, height):
        """
        Generate a forward flight trajectory.

        Args:
            num_points (int): Number of points in the trajectory.
            velocity (float): The fixed velocity of the drone.
            height (float): The fixed height (z) of the trajectory.

        Returns:
            list: A list of tuples, each containing (roll_torque, pitch_torque, yaw_torque, thrust).
        """
        reference_states = []
        t = np.linspace(0, num_points / velocity, num_points)  # Time array based on fixed velocity

        # Generate forward flight points
        x = velocity * t  # Forward motion in x
        y = np.zeros(num_points)  # Constant y position
        z = np.full(t.shape, height)  # Constant height

        # Assuming yaw is constant (facing forward)
        yaw = np.zeros(num_points)  # Constant yaw angle

        for i in range(num_points):
            control_inputs = self.compute_control_inputs((x[i], y[i], z[i]), yaw[i])
            reference_states.append(control_inputs)

        return reference_states, x, y, z, yaw

    def plot_trajectory(self, x, y, z):
        """
        Plot the trajectory in 3D.

        Args:
            x (ndarray): X coordinates of the trajectory.
            y (ndarray): Y coordinates of the trajectory.
            z (ndarray): Z coordinates of the trajectory.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory
        ax.plot(x, y, z, label='Trajectory', color='b')

        ax.set_title('Trajectory')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')

        # Manually set limits to ensure equal scaling
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    linearization_params = {
        'K_roll': 1.0,
        'K_pitch': 1.0,
        'K_yaw': 0.5,
        'K_thrust': 9.81,  # Thrust to maintain altitude
        'x_target': 0.0,
        'y_target': 0.0,
        'z_target': 1.0,  # Target altitude
        'yaw_target': 0.0  # Target yaw angle
    }

    trajectory_gen = Trajectory(linearization_parameters=linearization_params)

    # Generate and plot horizontal eight trajectory
    num_points = 100  # Number of points in the trajectory
    height = 1.0      # Constant height for the horizontal eight
    reference_states, x_eight, y_eight, z_eight, yaw_eight = trajectory_gen.generate_horizontal_eight_trajectory(num_points, height)

    print("Horizontal Eight Trajectory Reference States:")
    for i, state in enumerate(reference_states):
        print(f"State {i}: Roll Torque: {state[0]:.2f}, Pitch Torque: {state[1]:.2f}, Yaw Torque: {state[2]:.2f}, Thrust: {state[3]:.2f}")

    trajectory_gen.plot_trajectory(x_eight, y_eight, z_eight)

    # Generate and plot forward flight trajectory
    forward_velocity = 1.0  # Forward velocity in m/s
    reference_states_forward, x_forward, y_forward, z_forward, yaw_forward = trajectory_gen.generate_forward_flight_trajectory(num_points, forward_velocity, height)

    print("\nForward Flight Trajectory Reference States:")
    for i, state in enumerate(reference_states_forward):
        print(f"State {i}: Roll Torque: {state[0]:.2f}, Pitch Torque: {state[1]:.2f}, Yaw Torque: {state[2]:.2f}, Thrust: {state[3]:.2f}")

    trajectory_gen.plot_trajectory(x_forward, y_forward, z_forward)
