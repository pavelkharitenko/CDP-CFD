from uav import UAV
from controller import Controller
from path_planner import PathPlanner

def main():
    # Initialize UAV, Controller, and Path Planner
    Q = np.diag([1, 1, 1, 0.1, 0.1, 0.1, 10, 10, 10, 0.1, 0.1, 0.1])
    R = np.diag([0.01, 0.01, 0.01, 0.1])

    # Initialize the LQR controller
    lqr = LQRController(uav.A, uav.B, Q, R)

    # Initial state and UAV object
    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Example initial state
    uav = UAV(initial_state, mass=1.0, inertia=np.eye(3))

    planner = PathPlanner(start=[0, 0, 0], goal=[10, 10, 10])

    # Generate a smooth figure-eight trajectory
    figure_eight_waypoints = planner.generate_waypoints(trajectory_type="figure_eight", center=[0, 0, 1], radius=3)

    # Main simulation loop
    while True:

        # Get IMU data (current state)
        imu_data = get_imu_data_from_simulator()

        # Define the desired state (this could come from a path planner or target position)
        desired_state = [target_x, target_y, target_z, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Hover at target position

        # Compute the control input using LQR
        control_input = lqr.compute_control(imu_data, desired_state)

        # Update the UAV state with the control input and dynamics
        uav.update_state(control_input, dt=0.01)

if __name__ == "__main__":
    main()
