import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Planner:
    def __init__(self, start_position=(0, 0, 1), num_waypoints=100):
        """
        Initialize the Planner with a starting position and the number of waypoints.

        Args:
            start_position (tuple): Initial (x, y, z) position of the drone.
            num_waypoints (int): Number of waypoints to generate.
        """
        self.start_position = np.array(start_position)
        self.num_waypoints = num_waypoints

    def generate_straight_line(self, end_position, constant_yaw=False):
        """
        Generate waypoints for a straight-line trajectory.

        Args:
            end_position (tuple): The (x, y, z) coordinates of the end point.
            constant_yaw (bool): If True, keep the yaw constant. If False, calculate yaw based on direction of movement.

        Returns:
            list: A list of waypoints and corresponding yaw angles.
        """
        end_position = np.array(end_position)
        waypoints = np.linspace(self.start_position, end_position, self.num_waypoints)

        yaw_angles = []
        for i in range(1, len(waypoints)):
            delta_x = waypoints[i][0] - waypoints[i-1][0]
            delta_y = waypoints[i][1] - waypoints[i-1][1]
            yaw = np.arctan2(delta_y, delta_x) if not constant_yaw else 0.0
            yaw_angles.append(yaw)

        # Add the first yaw angle to match the first waypoint
        yaw_angles = [yaw_angles[0]] + yaw_angles
        return waypoints.tolist(), yaw_angles

    def generate_circle(self, radius=1.0, height=1.0, center=(0, 0), constant_yaw=False):
        """
        Generate waypoints for a circular trajectory.

        Args:
            radius (float): Radius of the circle.
            height (float): Height of the trajectory.
            center (tuple): The (x, y) coordinates of the center of the circle.
            constant_yaw (bool): If True, keep the yaw constant. If False, calculate yaw based on direction of movement.

        Returns:
            list: A list of waypoints and corresponding yaw angles.
        """
        center_x, center_y = center
        t = np.linspace(0, 2 * np.pi, self.num_waypoints)
        x = center_x + radius * np.cos(t)
        y = center_y + radius * np.sin(t)
        z = np.full_like(x, height)

        waypoints = np.column_stack((x, y, z))

        yaw_angles = []
        for i in range(1, len(waypoints)):
            delta_x = waypoints[i][0] - waypoints[i-1][0]
            delta_y = waypoints[i][1] - waypoints[i-1][1]
            yaw = np.arctan2(delta_y, delta_x) if not constant_yaw else 0.0
            yaw_angles.append(yaw)

        # Add the first yaw angle to match the first waypoint
        yaw_angles = [yaw_angles[0]] + yaw_angles
        return waypoints.tolist(), yaw_angles

    def generate_custom_path(self, path_points, constant_yaw=False):
        """
        Generate waypoints for a custom path based on provided points.

        Args:
            path_points (list): A list of (x, y, z) waypoints.
            constant_yaw (bool): If True, keep the yaw constant. If False, calculate yaw based on direction of movement.

        Returns:
            list: A list of waypoints and corresponding yaw angles.
        """
        waypoints = np.array(path_points)
        yaw_angles = []

        for i in range(1, len(waypoints)):
            delta_x = waypoints[i][0] - waypoints[i-1][0]
            delta_y = waypoints[i][1] - waypoints[i-1][1]
            yaw = np.arctan2(delta_y, delta_x) if not constant_yaw else 0.0
            yaw_angles.append(yaw)

        # Add the first yaw angle to match the first waypoint
        yaw_angles = [yaw_angles[0]] + yaw_angles
        return waypoints.tolist(), yaw_angles

    def plot_trajectory_with_yaw(self, waypoints, yaw_angles, scale=0.5):
        """
        Plot the waypoints and visualize the yaw orientation of the drone.

        Args:
            waypoints (list): List of (x, y, z) waypoints.
            yaw_angles (list): List of corresponding yaw angles.
            scale (float): Scale of the yaw vector to adjust its visual length.
        """
        waypoints = np.array(waypoints)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the waypoints as a trajectory
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], label='Trajectory')

        # Plot yaw vectors as small arrows
        for i, (wp, yaw) in enumerate(zip(waypoints, yaw_angles)):
            yaw_vector_x = scale * np.cos(yaw)
            yaw_vector_y = scale * np.sin(yaw)
            ax.quiver(wp[0], wp[1], wp[2], yaw_vector_x, yaw_vector_y, 0, color='r', label='Yaw' if i == 0 else '')

        # Labels and title
        ax.set_title("Trajectory with Yaw Orientation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set equal scale
        max_range = np.array([waypoints[:, 0].max() - waypoints[:, 0].min(), 
                              waypoints[:, 1].max() - waypoints[:, 1].min(), 
                              waypoints[:, 2].max() - waypoints[:, 2].min()]).max() / 2.0
        mid_x = (waypoints[:, 0].max() + waypoints[:, 0].min()) * 0.5
        mid_y = (waypoints[:, 1].max() + waypoints[:, 1].min()) * 0.5
        mid_z = (waypoints[:, 2].max() + waypoints[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.legend()
        plt.show()


# Example usage:
if __name__ == "__main__":
    planner = Planner(start_position=(0, 0, 1), num_waypoints=100)

    # Generate a straight-line path
    waypoints, yaw_angles = planner.generate_straight_line(end_position=(10, 0, 1))
    print("Straight-line waypoints and yaw angles:")
    for i, (wp, yaw) in enumerate(zip(waypoints, yaw_angles)):
        print(f"Waypoint {i}: {wp}, Yaw: {yaw:.2f} radians")

    # Plot the trajectory with yaw orientation
    planner.plot_trajectory_with_yaw(waypoints, yaw_angles)

    # Generate a circular path
    waypoints_circle, yaw_angles_circle = planner.generate_circle(radius=5, height=1)
    print("\nCircular path waypoints and yaw angles:")
    for i, (wp, yaw) in enumerate(zip(waypoints_circle, yaw_angles_circle)):
        print(f"Waypoint {i}: {wp}, Yaw: {yaw:.2f} radians")

    # Plot the circular trajectory with yaw orientation
    planner.plot_trajectory_with_yaw(waypoints_circle, yaw_angles_circle)
