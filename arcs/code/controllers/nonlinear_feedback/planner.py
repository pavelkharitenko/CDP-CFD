import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

class Planner:
    def __init__(self, start=(0.0, 0.0, 0.0), end=(5.0, 0.0, 0.0), 
                dt=0.1, velocity=0.3, acceleration_time=1.0, hover_time=0.0, 
                radius=2.5, traj_type=0, initial_yaw=0.0):
        self.start = np.array(start)
        self.end = np.array(end)
        self.dt = dt
        self.velocity = velocity
        self.acceleration_time = acceleration_time
        self.hover_time = hover_time
        self.radius = radius
        self.traj_type = traj_type
        self.current_index = 0
        self.initial_yaw = initial_yaw
        
        if self.traj_type == 0:
            self.trajectory = self._generate_linear_trajectory()
        elif self.traj_type == 1:
            self.trajectory = self._generate_circle_trajectory()
        elif self.traj_type == 2:
            self.trajectory = self._generate_horizontal_circle_trajectory()
        elif self.traj_type == 3:
            self.trajectory = self._generate_tilted_circle_trajectory_2()
        elif self.traj_type == 4:
            self.trajectory = self._generate_tilted_circle_trajectory_reversed()
        elif self.traj_type == 5:
            self.trajectory = self._generate_spiral_trajectory()
    
    def _generate_hover_waypoints(self):
        num_hover_points = int(self.hover_time / self.dt)
        hover_waypoints = [np.array([*self.start, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.initial_yaw]) 
                           for _ in range(num_hover_points)]
        return hover_waypoints

    def _smooth_acceleration_profile(self, total_time, num_waypoints):
        # Generate a linearly increasing acceleration from 0 to the required max value
        acceleration_points = int(self.acceleration_time / self.dt)
        acceleration = np.linspace(0, self.velocity / self.acceleration_time, acceleration_points)
        acceleration = np.concatenate((acceleration, np.full(num_waypoints - len(acceleration), 0.0)))
        return acceleration

    def _generate_linear_trajectory(self):
        direction = self.end - self.start
        distance = np.linalg.norm(direction)
        direction = direction / distance
        
        total_time = distance / self.velocity
        num_waypoints = int(total_time / self.dt) + 1
        
        # Get a smooth acceleration profile
        accelerations = self._smooth_acceleration_profile(total_time, num_waypoints)
        
        # Compute velocity as the integral of acceleration (cumulative sum of acceleration)
        velocities_magnitudes = np.cumsum(accelerations) * self.dt
        velocities_magnitudes = np.clip(velocities_magnitudes, 0, self.velocity)  # clip to max velocity
        
        # Create waypoints based on the velocity profile
        waypoints = [self.start + direction * sum(velocities_magnitudes[:i]) * self.dt for i in range(num_waypoints)]
        velocities = [vel_magnitude * direction for vel_magnitude in velocities_magnitudes]
        
        # Calculate accelerations (second derivative of position, or first derivative of velocity)
        accelerations = np.gradient(velocities, self.dt, axis=0)
        
        yaws = [self.initial_yaw for _ in range(num_waypoints)]
        
        # Combine position, velocity, acceleration, and yaw into the final trajectory
        trajectory = [np.array([*waypoint, *velocity, *acceleration, yaw]) 
                      for waypoint, velocity, acceleration, yaw in zip(waypoints, velocities, accelerations, yaws)]
        
        hover_waypoints = self._generate_hover_waypoints()
        full_trajectory = np.array(hover_waypoints + trajectory)
        
        return full_trajectory
    
    def _generate_circle_trajectory(self):
        circumference = 2 * np.pi * self.radius
        total_time = circumference / self.velocity
        num_waypoints = int(total_time / self.dt) + 1
        
        angles = np.linspace(0, 2 * np.pi, num_waypoints)
        angles -= np.pi / 2.0
        
        accelerations = self._smooth_acceleration_profile(total_time, num_waypoints)
        
        velocities_magnitudes = np.cumsum(accelerations) * self.dt
        velocities_magnitudes = np.clip(velocities_magnitudes, 0, self.velocity)  # clip to max velocity
        
        waypoints = [[self.start[0] + self.radius * np.cos(theta),
                       self.start[1] + self.radius * np.sin(theta) + self.radius,
                       self.start[2] + 0.0] for theta in angles]
        
        velocities = [[-vel_magnitude * np.sin(theta), vel_magnitude * np.cos(theta), 0.0] 
                      for vel_magnitude, theta in zip(velocities_magnitudes, angles)]
        
        accelerations = np.gradient(velocities, self.dt, axis=0)
        yaws = angles + np.pi / 2.0 + self.initial_yaw
        
        trajectory = [np.array([*waypoint, *velocity, *acceleration, yaw]) 
                      for waypoint, velocity, acceleration, yaw in zip(waypoints, velocities, accelerations, yaws)]
        
        hover_waypoints = self._generate_hover_waypoints()
        full_trajectory = np.array(hover_waypoints + trajectory)
        
        return full_trajectory
    
    def _generate_horizontal_circle_trajectory(self):
        circumference = 2 * np.pi * self.radius
        total_time = circumference / self.velocity
        num_waypoints = int(total_time / self.dt) + 1
        
        angles = np.linspace(0, 2 * np.pi, num_waypoints)
        
        accelerations = self._smooth_acceleration_profile(total_time, num_waypoints)
        
        velocities_magnitudes = np.cumsum(accelerations) * self.dt
        velocities_magnitudes = np.clip(velocities_magnitudes, 0, self.velocity)  # clip to max velocity
        
        waypoints = [[0.0 + self.start[0], 
                      self.radius * np.cos(theta) + self.start[1]  - self.radius, 
                      self.radius * np.sin(theta) + self.start[2]] 
                      for theta in angles]
        
        velocities = [[0.0, -vel_magnitude * np.sin(theta), vel_magnitude * np.cos(theta)] 
                      for vel_magnitude, theta in zip(velocities_magnitudes, angles)]
        
        accelerations = np.gradient(velocities, self.dt, axis=0)
        yaws = [self.initial_yaw for _ in range(num_waypoints)]
        
        trajectory = [np.array([*waypoint, *velocity, *acceleration, yaw]) 
                      for waypoint, velocity, acceleration, yaw in zip(waypoints, velocities, accelerations, yaws)]
        
        hover_waypoints = self._generate_hover_waypoints()
        full_trajectory = np.array(hover_waypoints + trajectory)
        
        return full_trajectory
   
    
    


    def _generate_tilted_circle_trajectory(self):
        # Define waypoints relative to the starting position
        relative_waypoints = [
            np.array([0, 0, 0]),
            np.array([-1.5, 1.5, -0.75]),
            np.array([-3, 0, 0]),
            np.array([-1.5, -1.5, 0.75]),
            np.array([0, 0, 0])  # Complete the loop back to start
        ]
        
        # Offset the waypoints by the starting position
        waypoints = np.array([self.start + wp for wp in relative_waypoints])
        
        # Number of points to interpolate along the curve
        total_points = int(len(waypoints) * 1 / self.dt)
        
        # Generate smooth spline for x, y, z
        t_waypoints = np.linspace(0, 1, len(waypoints))  # Parameter for the waypoints
        t_spline = np.linspace(0, 1, total_points)      # Fine-grained parameter for the trajectory
        
        spline_x = CubicSpline(t_waypoints, waypoints[:, 0], bc_type='periodic')
        spline_y = CubicSpline(t_waypoints, waypoints[:, 1], bc_type='periodic')
        spline_z = CubicSpline(t_waypoints, waypoints[:, 2], bc_type='periodic')
        
        # Generate smooth positions
        smooth_positions = np.stack([spline_x(t_spline), spline_y(t_spline), spline_z(t_spline)], axis=-1)
        
        # Compute velocities and accelerations from the splines
        raw_velocities = np.stack([spline_x(t_spline, 1), spline_y(t_spline, 1), spline_z(t_spline, 1)], axis=-1)
        raw_accelerations = np.stack([spline_x(t_spline, 2), spline_y(t_spline, 2), spline_z(t_spline, 2)], axis=-1)
        
        # Apply smoothing to velocities and accelerations
        smooth_velocities = savgol_filter(raw_velocities, window_length=51, polyorder=3, axis=0)  # Adjust window length
        smooth_accelerations = savgol_filter(raw_accelerations, window_length=51, polyorder=3, axis=0)

        # Clip velocities to ensure they don't exceed the desired maximum velocity
        velocity_magnitudes = np.linalg.norm(smooth_velocities, axis=1)
        scale_factors = np.minimum(1, self.velocity / np.maximum(velocity_magnitudes, 1e-5))
        smooth_velocities = (smooth_velocities.T * scale_factors).T
        
        # Recalculate accelerations to match adjusted velocities
        smooth_accelerations = np.gradient(smooth_velocities, self.dt, axis=0)
        
        # Compute yaw angles based on forward direction
        forward_directions = np.diff(smooth_positions, axis=0, prepend=smooth_positions[0:1])
        yaw_angles = np.arctan2(forward_directions[:, 1], forward_directions[:, 0])
        
        # Combine position, velocity, acceleration, and yaw into the trajectory
        trajectory = [
            np.array([*pos, *vel, *acc, yaw])
            for pos, vel, acc, yaw in zip(smooth_positions, smooth_velocities, smooth_accelerations, yaw_angles)
        ]
        
        hover_waypoints = self._generate_hover_waypoints()
        return np.array(hover_waypoints + trajectory)


    def _generate_tilted_circle_trajectory_reversed(self):
        # Define waypoints relative to the starting position
        relative_waypoints = [
            np.array([0, 0, 0]),
            np.array([1.5, 1.5, 0.75]),
            np.array([3, 0, 0]),
            np.array([1.5, -1.5, -0.75]),
            np.array([0, 0, 0])  # Complete the loop back to start
        ]
        
        # Offset the waypoints by the starting position
        waypoints = np.array([self.start + wp for wp in relative_waypoints])
        
        # Number of points to interpolate along the curve
        total_points = int(len(waypoints) * 1 / self.dt)
        
        # Generate smooth spline for x, y, z
        t_waypoints = np.linspace(0, 1, len(waypoints))  # Parameter for the waypoints
        t_spline = np.linspace(0, 1, total_points)      # Fine-grained parameter for the trajectory
        
        spline_x = CubicSpline(t_waypoints, waypoints[:, 0], bc_type='periodic')
        spline_y = CubicSpline(t_waypoints, waypoints[:, 1], bc_type='periodic')
        spline_z = CubicSpline(t_waypoints, waypoints[:, 2], bc_type='periodic')
        
        # Generate smooth positions
        smooth_positions = np.stack([spline_x(t_spline), spline_y(t_spline), spline_z(t_spline)], axis=-1)
        
        # Compute velocities and accelerations from the splines
        raw_velocities = np.stack([spline_x(t_spline, 1), spline_y(t_spline, 1), spline_z(t_spline, 1)], axis=-1)
        raw_accelerations = np.stack([spline_x(t_spline, 2), spline_y(t_spline, 2), spline_z(t_spline, 2)], axis=-1)
        
        # Apply smoothing to velocities and accelerations
        smooth_velocities = savgol_filter(raw_velocities, window_length=51, polyorder=3, axis=0)  # Adjust window length
        smooth_accelerations = savgol_filter(raw_accelerations, window_length=51, polyorder=3, axis=0)

        # Clip velocities to ensure they don't exceed the desired maximum velocity
        velocity_magnitudes = np.linalg.norm(smooth_velocities, axis=1)
        scale_factors = np.minimum(1, self.velocity / np.maximum(velocity_magnitudes, 1e-5))
        smooth_velocities = (smooth_velocities.T * scale_factors).T
        
        # Recalculate accelerations to match adjusted velocities
        smooth_accelerations = np.gradient(smooth_velocities, self.dt, axis=0)
        
        # Compute yaw angles based on forward direction
        forward_directions = np.diff(smooth_positions, axis=0, prepend=smooth_positions[0:1])
        yaw_angles = np.arctan2(forward_directions[:, 1], forward_directions[:, 0])
        
        # Combine position, velocity, acceleration, and yaw into the trajectory
        trajectory = [
            np.array([*pos, *vel, *acc, yaw])
            for pos, vel, acc, yaw in zip(smooth_positions, smooth_velocities, smooth_accelerations, yaw_angles)
        ]
        
        hover_waypoints = self._generate_hover_waypoints()
        return np.array(hover_waypoints + trajectory)


    def _generate_tilted_circle_trajectory_2(self):
        # Define waypoints relative to the starting position
        relative_waypoints = [
            np.array([0, 0, 0]),
            np.array([-1.5, 1.5, -0.75]),
            np.array([-3, 0, 0]),
            np.array([-1.5, -1.5, 0.75]),
            np.array([0, 0, 0])  # Complete the loop back to start
        ]
        
        # Offset the waypoints by the starting position
        waypoints = np.array([self.start + wp for wp in relative_waypoints])
        
        # Number of points to interpolate along the curve
        total_points = int(len(waypoints) * 1 / self.dt)
        
        # Generate smooth spline for x, y, z
        t_waypoints = np.linspace(0, 1, len(waypoints))  # Parameter for the waypoints
        t_spline = np.linspace(0, 1, total_points)      # Fine-grained parameter for the trajectory
        
        spline_x = CubicSpline(t_waypoints, waypoints[:, 0], bc_type='periodic')
        spline_y = CubicSpline(t_waypoints, waypoints[:, 1], bc_type='periodic')
        spline_z = CubicSpline(t_waypoints, waypoints[:, 2], bc_type='periodic')
        
        # Generate smooth positions
        smooth_positions = np.stack([spline_x(t_spline), spline_y(t_spline), spline_z(t_spline)], axis=-1)
        
        # Compute velocities and accelerations from the splines
        raw_velocities = np.stack([spline_x(t_spline, 1), spline_y(t_spline, 1), spline_z(t_spline, 1)], axis=-1)
        raw_accelerations = np.stack([spline_x(t_spline, 2), spline_y(t_spline, 2), spline_z(t_spline, 2)], axis=-1)
        
        # Apply Savitzky-Golay filter to smooth velocities and accelerations
        smooth_velocities = savgol_filter(raw_velocities, window_length=51, polyorder=3, axis=0)
        smooth_accelerations = savgol_filter(raw_accelerations, window_length=51, polyorder=3, axis=0)

        # Clip velocities to ensure they don't exceed the desired maximum velocity
        velocity_magnitudes = np.linalg.norm(smooth_velocities, axis=1)
        scale_factors = np.minimum(1, self.velocity / np.maximum(velocity_magnitudes, 1e-5))
        smooth_velocities = (smooth_velocities.T * scale_factors).T
        
        # Recalculate accelerations to match adjusted velocities
        smooth_accelerations = np.gradient(smooth_velocities, self.dt, axis=0)
        
        # Compute yaw angles based on forward direction
        forward_directions = np.diff(smooth_positions, axis=0, prepend=smooth_positions[0:1])
        yaw_angles = np.arctan2(forward_directions[:, 1], forward_directions[:, 0])
        
        # Combine position, velocity, acceleration, and yaw into the trajectory
        trajectory = [
            np.array([*pos, *vel, *acc, yaw])
            for pos, vel, acc, yaw in zip(smooth_positions, smooth_velocities, smooth_accelerations, yaw_angles)
        ]
        
        # Generate hover waypoints for the trajectory
        hover_waypoints = self._generate_hover_waypoints()
        return np.array(hover_waypoints + trajectory)


    def _generate_spiral_trajectory(self):
        # Define waypoints relative to the starting position
        relative_waypoints = [
            np.array([0, 0, 0]),
            np.array([-2.0, 2.0, -1.0]),
            np.array([-4, 0, 0]),
            np.array([-2.0, -2.0, 0.0]),
            np.array([0, 0, 0]),  # Complete the loop back to start
            np.array([-2.0, 2.0, 1.0]),
            np.array([-4, 0, 0]),
            np.array([-2.0, -2.0, 0.0]),
            np.array([0, 0, 0])
        ]
        
        # Offset the waypoints by the starting position
        waypoints = np.array([self.start + wp for wp in relative_waypoints])
        
        # Number of points to interpolate along the curve
        total_points = int(len(waypoints) * 1 / self.dt)
        
        # Generate smooth spline for x, y, z
        t_waypoints = np.linspace(0, 1, len(waypoints))  # Parameter for the waypoints
        t_spline = np.linspace(0, 1, total_points)      # Fine-grained parameter for the trajectory
        
        spline_x = CubicSpline(t_waypoints, waypoints[:, 0], bc_type='periodic')
        spline_y = CubicSpline(t_waypoints, waypoints[:, 1], bc_type='periodic')
        spline_z = CubicSpline(t_waypoints, waypoints[:, 2], bc_type='periodic')
        
        # Generate smooth positions
        smooth_positions = np.stack([spline_x(t_spline), spline_y(t_spline), spline_z(t_spline)], axis=-1)
        
        # Compute velocities and accelerations from the splines
        raw_velocities = np.stack([spline_x(t_spline, 1), spline_y(t_spline, 1), spline_z(t_spline, 1)], axis=-1)
        raw_accelerations = np.stack([spline_x(t_spline, 2), spline_y(t_spline, 2), spline_z(t_spline, 2)], axis=-1)
        
        # Apply Savitzky-Golay filter to smooth velocities and accelerations
        smooth_velocities = savgol_filter(raw_velocities, window_length=51, polyorder=3, axis=0)
        smooth_accelerations = savgol_filter(raw_accelerations, window_length=51, polyorder=3, axis=0)

        # Clip velocities to ensure they don't exceed the desired maximum velocity
        velocity_magnitudes = np.linalg.norm(smooth_velocities, axis=1)
        scale_factors = np.minimum(1, self.velocity / np.maximum(velocity_magnitudes, 1e-5))
        smooth_velocities = (smooth_velocities.T * scale_factors).T
        
        # Recalculate accelerations to match adjusted velocities
        smooth_accelerations = np.gradient(smooth_velocities, self.dt, axis=0)
        
        # Compute yaw angles based on forward direction
        forward_directions = np.diff(smooth_positions, axis=0, prepend=smooth_positions[0:1])
        yaw_angles = np.arctan2(forward_directions[:, 1], forward_directions[:, 0])
        
        # Combine position, velocity, acceleration, and yaw into the trajectory
        trajectory = [
            np.array([*pos, *vel, *acc, yaw])
            for pos, vel, acc, yaw in zip(smooth_positions, smooth_velocities, smooth_accelerations, yaw_angles)
        ]
        
        # Generate hover waypoints for the trajectory
        hover_waypoints = self._generate_hover_waypoints()
        return np.array(hover_waypoints + trajectory)




    def adjust_waypoint(self, current_state, next_waypoint, alpha=0.9):
        current_position = current_state[:3]
        current_velocity = current_state[3:6]
        current_acceleration = current_state[6:9]
        
        desired_position = next_waypoint[:3]
        desired_velocity = next_waypoint[3:6]
        desired_acceleration = next_waypoint[6:9]
        desired_yaw = next_waypoint[9]
        
        adjusted_velocity = current_velocity + alpha * (desired_velocity - current_velocity)
        adjusted_acceleration = current_acceleration + alpha * (desired_acceleration - current_acceleration)
        
        adjusted_waypoint = np.array([*desired_position, *adjusted_velocity, *adjusted_acceleration, desired_yaw])
        
        return adjusted_waypoint
    
    def pop_waypoint(self, current_state, alpha=1.0):
        if self.current_index < len(self.trajectory):
            next_waypoint = self.trajectory[self.current_index]
            adjusted_waypoint = self.adjust_waypoint(current_state, next_waypoint, alpha)
            self.current_index += 1
            return adjusted_waypoint
        else:
            if self.traj_type == 0:
                return self.trajectory[-1]
            else:
                self.current_index = 0
                return self.pop_waypoint(current_state)

                
    
    def plot_trajectory(self):
        waypoints = self.trajectory[:, :3]
        velocities = self.trajectory[:, 3:6]
        accelerations = self.trajectory[:, 6:9]
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='black', label='Waypoints')
        ax.quiver(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                  velocities[:, 0], velocities[:, 1], velocities[:, 2], 
                  color='blue', label='Velocities')
        ax.quiver(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                  accelerations[:, 0], accelerations[:, 1], accelerations[:, 2], 
                  color='red', label='Accelerations')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory with Velocity and Acceleration Vectors')
        ax.legend()
        
        plt.show()



    def plot_trajectory_2d(self):
        # Extract position, velocity, and acceleration data
        waypoints = self.trajectory[:, :3]  # Positions (x, y, z)
        velocities = self.trajectory[:, 3:6]  # Velocities (vx, vy, vz)
        accelerations = self.trajectory[:, 6:9]  # Accelerations (ax, ay, az)
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot positions
        axs[0].plot(waypoints[:, 0], label='X Position', color='black')
        axs[0].plot(waypoints[:, 1], label='Y Position', color='blue')
        axs[0].plot(waypoints[:, 2], label='Z Position', color='green')
        axs[0].set_ylabel('Position')
        axs[0].set_title('Position vs Time')
        axs[0].legend()
        
        # Plot velocities
        axs[1].plot(velocities[:, 0], label='X Velocity', color='blue')
        axs[1].plot(velocities[:, 1], label='Y Velocity', color='orange')
        axs[1].plot(velocities[:, 2], label='Z Velocity', color='red')
        axs[1].set_ylabel('Velocity')
        axs[1].set_title('Velocity vs Time')
        axs[1].legend()

        # Plot accelerations
        axs[2].plot(accelerations[:, 0], label='X Acceleration', color='red')
        axs[2].plot(accelerations[:, 1], label='Y Acceleration', color='green')
        axs[2].plot(accelerations[:, 2], label='Z Acceleration', color='purple')
        axs[2].set_ylabel('Acceleration')
        axs[2].set_title('Acceleration vs Time')
        axs[2].legend()

        axs[2].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

if False:
    planner_linear = Planner(hover_time=2.0)
    planner_linear.plot_trajectory()
    planner_circular = Planner(circular=True, hover_time=2.0)
    planner_circular.plot_trajectory()

    while True:
        waypoint = planner_circular.pop_waypoint()
        if waypoint is None:
            break
        print(f"Waypoint: {waypoint}")
