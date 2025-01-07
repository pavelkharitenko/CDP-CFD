import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Planner:
    def __init__(self, start=(0.0, 0.0, 0.0), end=(5.0, 0.0, 0.0), 
                 step_size=0.1, velocity=0.3, acceleration_time=1.0, hovertime=0.0, 
                 radius=2.0, circular=False, initial_yaw=0.0):
        self.start = np.array(start)
        self.end = np.array(end)
        self.step_size = step_size
        self.velocity = velocity
        self.acceleration_time = acceleration_time  # Time to reach target velocity
        self.radius = radius
        self.circular = circular
        self.current_index = 0
        self.initial_yaw = initial_yaw
        
        if self.circular:
            self.trajectory = self._generate_circle_trajectory()
        else:
            self.trajectory = self._generate_linear_trajectory()
    
    def _linear_velocity_profile(self, total_time, num_waypoints):
        velocities = np.linspace(0, self.velocity, int(self.acceleration_time / self.step_size))
        velocities = np.concatenate((velocities, np.full(num_waypoints - len(velocities), self.velocity)))
        return velocities
    
    def _generate_linear_trajectory(self):
        direction = self.end - self.start
        distance = np.linalg.norm(direction)
        direction = direction / distance  # Normalize direction vector
        
        total_time = distance / self.velocity
        num_waypoints = int(total_time / self.step_size) + 1
        
        velocities_magnitudes = self._linear_velocity_profile(total_time, num_waypoints)
        
        waypoints = [self.start + direction * sum(velocities_magnitudes[:i]) * self.step_size for i in range(num_waypoints)]
        velocities = [vel_magnitude * direction for vel_magnitude in velocities_magnitudes]
        accelerations = np.gradient(velocities, axis=0) / self.step_size
        yaws = [self.initial_yaw for _ in range(num_waypoints)]  # keep yaw as is for linear trajectory
        
        trajectory = [np.array([*waypoint, *velocity, *acceleration, yaw]) 
                      for waypoint, velocity, acceleration, yaw in zip(waypoints, velocities, accelerations, yaws)]
        
        return np.array(trajectory)
    
    def _generate_circle_trajectory(self):
        circumference = 2 * np.pi * self.radius
        total_time = circumference / self.velocity
        num_waypoints = int(total_time / self.step_size) + 1
        
        angles = np.linspace(0, 2 * np.pi, num_waypoints)
        angles -= np.pi/2.0
        velocities_magnitudes = self._linear_velocity_profile(total_time, num_waypoints)
        
        # Shift circle to start at (0,0,0) and follow a counterclockwise path
        waypoints = [[self.radius * np.cos(theta), self.radius * np.sin(theta) + self.radius, 0.0] for theta in angles]
        velocities = [[-vel_magnitude * np.sin(theta), vel_magnitude * np.cos(theta), 0.0] 
                      for vel_magnitude, theta in zip(velocities_magnitudes, angles)]
        accelerations = np.gradient(velocities, axis=0) / self.step_size
        yaws = angles + np.pi/2.0  # Yaw follows the tangent direction to the circle
        
        trajectory = [np.array([*waypoint, *velocity, *acceleration, yaw]) 
                      for waypoint, velocity, acceleration, yaw in zip(waypoints, velocities, accelerations, yaws)]
        
        return np.array(trajectory)
    
    def pop_waypoint(self):
        if self.current_index < len(self.trajectory):
            waypoint = self.trajectory[self.current_index]
            self.current_index += 1
            return waypoint
        else:
            return None
    
    def plot_trajectory(self):
        waypoints = self.trajectory[:, :3]
        velocities = self.trajectory[:, 3:6]
        accelerations = self.trajectory[:, 6:9]
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot waypoints
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='black', label='Waypoints')
        
        # Plot velocity vectors
        ax.quiver(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                  velocities[:, 0], velocities[:, 1], velocities[:, 2], 
                  color='blue', label='Velocities')
        
        # Plot acceleration vectors
        ax.quiver(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                  accelerations[:, 0], accelerations[:, 1], accelerations[:, 2], 
                  color='red', label='Accelerations')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory Preview')
        ax.legend()
        plt.show()

if False:
    # Example usage for circular trajectory
    planner_linear = Planner()
    planner_linear.plot_trajectory()
    planner_circular = Planner(circular=True)
    planner_circular.plot_trajectory()

    while True:
        waypoint = planner_circular.pop_waypoint()
        if waypoint is None:
            break
        print(f"Waypoint: {waypoint}")
