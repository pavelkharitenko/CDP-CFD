import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, A, B, Q=np.eye(7)*0.5, R=np.eye(4)*2):
        """
        Initialize the LQR controller.

        Args:
            A (np.array): State transition matrix.
            B (np.array): Control input matrix.
            Q (np.array): State cost matrix.
            R (np.array): Control cost matrix.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        # Solve for the optimal gain matrix K
        self.K = self.compute_lqr_gain()

    def compute_lqr_gain(self):
        """
        Compute the LQR gain matrix K using the solution of the continuous-time algebraic Riccati equation.
        
        Returns:
            np.array: The optimal gain matrix K.
        """
        # Solve the continuous-time algebraic Riccati equation (ARE)
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        
        # Compute the LQR gain K
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K

    def control_input(self, state):
        """
        Compute the control input using the state feedback law u = -Kx.
        
        Args:
            state (np.array): The current state vector x.
        
        Returns:
            np.array: The control input vector u.
        """
        # Apply the LQR feedback law
        u = -self.K @ state
        return u
