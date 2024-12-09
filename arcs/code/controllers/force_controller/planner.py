import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_circular_path(radius, num_points, height=1.0):
    """
    Generate a set of 3D points and yaw angles for a quadrotor to follow in a circular path.

    Parameters:
    radius (float): Radius of the circle.
    num_points (int): Number of points in the path.
    height (float): Height of the circular path (default: 1m).

    Returns:
    points (np.ndarray): Array of shape (num_points, 4) with columns [x, y, z, yaw].
    """
    # Generate the circle in the XY plane
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, height)

    # Calculate yaw angle at each point (aligned with the tangent to the circle)
    yaw = np.arctan2(np.diff(y, append=y[0]), np.diff(x, append=x[0]))

    # Stack coordinates and yaw angles into a single array
    points = np.vstack((x, y, z, yaw)).T
    return points

def plot_circular_path(points):
    """
    Plot the 3D circular path with yaw vectors.

    Parameters:
    points (np.ndarray): Array of shape (num_points, 4) with columns [x, y, z, yaw].
    """
    x, y, z, yaw = points.T

    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the circular path
    ax.plot(x, y, z, label='Circular Path', color='blue')
    ax.scatter(x, y, z, color='red', s=10, label='Waypoints')

    # Add yaw vectors
    u = np.cos(yaw)
    v = np.sin(yaw)
    w = np.zeros_like(yaw)
    ax.quiver(x, y, z, u, v, w, length=0.2, normalize=True, color='green', label='Yaw Vectors')

    # Set labels and aspect
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Circular Path with Yaw Vectors')
    ax.legend()
    ax.grid(True)

    # Show the plot
    plt.show()

# Example usage
radius = 1.0
num_points = 100
path_points = generate_circular_path(radius, num_points)
plot_circular_path(path_points)
