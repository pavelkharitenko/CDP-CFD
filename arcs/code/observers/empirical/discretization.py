import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


def discretize_shapes(vertices_list, plot=False):
    """
    Creates 2d frame from vertices, disretizes and plots optionally gridpoints
    """
    grid_size = 0.03
    total_grid_points = []

    if plot:
            plt.figure(figsize=(8, 8))

    for vertices in vertices_list:
        frame_shape = Polygon(vertices)

        # Set the grid size (5 cm = 0.05 meters)

        # Define the bounding box for the grid based on the diamond shape
        min_x, min_y, max_x, max_y = frame_shape.bounds

        # Generate the grid points within the bounding box
        x_coords = np.arange(min_x, max_x, grid_size)
        y_coords = np.arange(min_y, max_y, grid_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        # Check which points are inside the diamond shape
        inside_points = [point for point in grid_points if frame_shape.contains(Point(point))]

        # Convert the inside points to an array for plotting
        inside_points = np.array(inside_points)
        print("inside p")
        print(inside_points)

        # Plot the diamond shape and the discretized points
        if plot:
            #plt.figure(figsize=(8, 8))
            plt.plot(*zip(*vertices, vertices[0]), color='black', linewidth=2)
            plt.scatter(inside_points[:, 0], inside_points[:, 1], color='blue', s=4)
    if plot:
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.legend()
        plt.title(f"Discretized Frame for {grid_size}m grid cell length")
        plt.axis("equal")
        plt.grid(True)
        plt.show()
    
    return 


body_frame_vertices = [
    (-0.03, 0.1305), (0.03, 0.1305), 
                        (0.0865, 0.0735), 
                        (0.0865, 0.0575), 
                    (0.07, 0.0375), 
                    (0.07, -0.0375),
                        (0.0865,  -0.0575),
                        (0.0865,  -0.0735),
                (0.026, -0.1305),
                (-0.026, -0.1305),
                        (-0.0865,  -0.0735),
                        (-0.0865,  -0.0575),
                    (-0.07, -0.0375),
                    (-0.07, 0.0375), 
                        (-0.0865, 0.0575), 
                (-0.0865, 0.0735),           
                    ]

arm_ur_vertices = np.array([(0.0865, 0.0735), (0.203173, 0.190173), (0.210244, 0.183102),
                (0.238528,0.211386), (0.210244, 0.239670), (0.181960, 0.211386),
                (0.189031,0.204315), (0.0723584,0.0876424), ])

arm_ll_vertices = -arm_ur_vertices
arm_ul_vertices = np.array([(-vert[0], vert[1])for vert in arm_ur_vertices])
arm_lr_vertices = -arm_ul_vertices






discretize_shapes([body_frame_vertices, arm_ur_vertices, arm_ll_vertices, arm_ul_vertices, arm_lr_vertices], True)