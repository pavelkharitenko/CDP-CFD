import numpy as np
import matplotlib.pyplot as plt

class SimplifiedPlanner:
    def __init__(self, start=(0.0, 0.0, 0.0), end=(5.0, 0.0, 0.0), dt=0.1, velocity=0.3, hover_time=0.0, target_yaw=0.0, traj_type=0):
        self.start = np.array(start)
        self.end = np.array(end)
        self.dt = dt
        self.velocity = velocity
        self.hover_time = hover_time
        self.target_yaw = target_yaw
        self.current_index = 0
        self.traj_type = traj_type
        
        

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
        hover_waypoints = [np.array([*self.start, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.target_yaw]) for _ in range(num_hover_points)]
        return hover_waypoints


    def _generate_linear_trajectory(self):
        direction = self.end - self.start
        distance = np.linalg.norm(direction)
        direction = direction / distance  # Normalize direction vector

        # Time to reach desired velocity (v / 2.0 seconds)
        acceleration_time = self.velocity * 0.75

        # Distance covered during acceleration phase
        acceleration_distance = 0.5 * self.velocity * acceleration_time

        # Distance covered at constant velocity
        constant_velocity_distance = distance - acceleration_distance

        # Time spent at constant velocity
        constant_velocity_time = constant_velocity_distance / self.velocity

        # Total time = acceleration time + constant velocity time
        total_time = acceleration_time + constant_velocity_time

        # Number of waypoints
        num_waypoints = int(total_time / self.dt) + 1

        # Generate time array
        times = np.linspace(0, total_time, num_waypoints)

        # Velocity profile: quadratic increase during acceleration, then constant
        velocities_magnitudes = np.where(
            times <= acceleration_time,
            self.velocity * (times / acceleration_time) ** 2,  # Quadratic increase
            self.velocity  # Constant velocity
        )

        # Compute accelerations as the derivative of velocities
        accelerations_magnitudes = np.gradient(velocities_magnitudes, self.dt)

        # Create waypoints with position, velocity, acceleration, and yaw
        waypoints = []
        current_position = self.start
        for i in range(num_waypoints):
            # Update position using velocity and direction
            current_position = current_position + direction * velocities_magnitudes[i] * self.dt
            
            # Compute velocity and acceleration vectors
            velocity_vector = direction * velocities_magnitudes[i]
            acceleration_vector = direction * accelerations_magnitudes[i]
            
            # Append the waypoint (position, velocity, acceleration, yaw)
            waypoints.append(np.array([*current_position, *velocity_vector, *acceleration_vector, self.target_yaw]))

        hover_waypoints = self._generate_hover_waypoints()
        full_trajectory = np.array(hover_waypoints + waypoints)

        return full_trajectory


    def _generate_circle_trajectory(self):
        circumference = 2 * np.pi * self.radius
        total_time = circumference / self.velocity
        num_waypoints = int(total_time / self.dt) + 1
        
        angles = np.linspace(0, 2 * np.pi, num_waypoints)
        angles -= np.pi / 2.0  # Adjust starting angle
        
        # Velocity profile: quadratic increase during acceleration, then constant
        acceleration_time = self.velocity / 2.0
        times = np.linspace(0, total_time, num_waypoints)
        velocities_magnitudes = np.where(
            times <= acceleration_time,
            self.velocity * (times / acceleration_time) ** 2,  # Quadratic increase
            self.velocity  # Constant velocity
        )

        # Compute accelerations as the derivative of velocities
        accelerations_magnitudes = np.gradient(velocities_magnitudes, self.dt)

        # Generate waypoints
        waypoints = [[self.start[0] + self.radius * np.cos(theta),
                      self.start[1] + self.radius * np.sin(theta) + self.radius,
                      self.start[2] + 0.0] for theta in angles]
        
        # Compute velocities and accelerations
        velocities = [[-vel_magnitude * np.sin(theta), vel_magnitude * np.cos(theta), 0.0] 
                      for vel_magnitude, theta in zip(velocities_magnitudes, angles)]
        accelerations = np.gradient(velocities, self.dt, axis=0)
        
        # Yaw angles (tangential to the circle)
        yaws = angles + np.pi / 2.0 + self.target_yaw
        
        # Combine into trajectory
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
        
        # Velocity profile: quadratic increase during acceleration, then constant
        acceleration_time = self.velocity / 2.0
        times = np.linspace(0, total_time, num_waypoints)
        velocities_magnitudes = np.where(
            times <= acceleration_time,
            self.velocity * (times / acceleration_time) ** 2,  # Quadratic increase
            self.velocity  # Constant velocity
        )

        # Compute accelerations as the derivative of velocities
        accelerations_magnitudes = np.gradient(velocities_magnitudes, self.dt)

        # Generate waypoints
        waypoints = [[0.0 + self.start[0], 
                      self.radius * np.cos(theta) + self.start[1] - self.radius, 
                      self.radius * np.sin(theta) + self.start[2]] 
                      for theta in angles]
        
        # Compute velocities and accelerations
        velocities = [[0.0, -vel_magnitude * np.sin(theta), vel_magnitude * np.cos(theta)] 
                      for vel_magnitude, theta in zip(velocities_magnitudes, angles)]
        accelerations = np.gradient(velocities, self.dt, axis=0)
        
        # Yaw angles (constant for horizontal circle)
        yaws = [self.target_yaw for _ in range(num_waypoints)]
        
        # Combine into trajectory
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
        
        # Apply smoothing to velocities and accelerations
        smooth_velocities = savgol_filter(raw_velocities, window_length=51, polyorder=3, axis=0)
        smooth_accelerations = savgol_filter(raw_accelerations, window_length=51, polyorder=3, axis=0)



    def pop_waypoint(self):
        if self.current_index < len(self.trajectory):
            next_waypoint = self.trajectory[self.current_index]
            self.current_index += 1
            return next_waypoint
        else:
            return self.trajectory[-1]

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
        waypoints = self.trajectory[:, :3]  # Positions (x, y, z)
        velocities = self.trajectory[:, 3:6]  # Velocities (vx, vy, vz)
        accelerations = self.trajectory[:, 6:9]  # Accelerations (ax, ay, az)
        yaws = self.trajectory[:, 9]  # Yaws
        
        fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
        
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

        # Plot yaw
        axs[3].plot(yaws, label='Yaw', color='brown')
        axs[3].set_ylabel('Yaw (rad)')
        axs[3].set_title('Yaw vs Time')
        axs[3].legend()

        axs[3].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

# Example usage
#planner = SimplifiedPlanner(start=(0.0, -2.5, 0.0), end=(0.0, 7.0, 0.0), dt=0.1, velocity=1.0, target_yaw=np.pi/4)
#planner.plot_trajectory_2d()