import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, A, B, Q, R):
        """
        Initialize the LQR controller.
        A: System dynamics matrix (linearized)
        B: Control input matrix (linearized)
        Q: State cost matrix
        R: Control input cost matrix
        """
        # Solve the Continuous Algebraic Riccati Equation (CARE)
        P = solve_continuous_are(A, B, Q, R)
        # Compute the LQR gain matrix
        self.K = np.linalg.inv(R).dot(B.T).dot(P)
    
    def compute_control(self, state, desired_state):
        """
        Compute the control input using the LQR feedback law.
        state: The current state of the UAV
        desired_state: The desired state of the UAV
        Returns: control_input vector [roll_torque, pitch_torque, yaw_torque, throttle]
        """
        state_error = state - desired_state
        control_input = -self.K.dot(state_error)
        return control_input
