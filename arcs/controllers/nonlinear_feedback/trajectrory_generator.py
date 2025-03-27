import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MinimumSnapTrajectory:
    def __init__(self, waypoints, average_velocity, dt=0.1):
        self.waypoints = np.array(waypoints)
        self.average_velocity = average_velocity
        self.dt = dt
        self.times = self._compute_times()
        self.coefficients = self._compute_trajectory()

    def _compute_times(self):
        distances = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        times = distances / self.average_velocity
        cumulative_times = np.insert(np.cumsum(times), 0, 0)
        return cumulative_times

    def _polynomial_cost(self, coeffs):
        total_cost = 0.0
        for i in range(len(self.times) - 1):
            T = self.times[i+1] - self.times[i]
            # Snap cost is associated with the 4th, 5th, 6th, and 7th degree terms
            snap_cost = (coeffs[7*i+4]**2 * 24**2 +
                         coeffs[7*i+5]**2 * 120**2 * T**2 +
                         coeffs[7*i+6]**2 * 360**2 * T**4 +
                         coeffs[7*i+7]**2 * 840**2 * T**6)
            total_cost += snap_cost * T  # Integrate over the segment time
        return total_cost

    def _compute_trajectory(self):
        num_segments = len(self.waypoints) - 1
        num_coeffs = num_segments * 8  # 8 coefficients per segment for a 7th-degree polynomial
        initial_guess = np.zeros(num_coeffs)

        # Constraints to ensure the trajectory passes through waypoints with continuity in position
        def constraint_eq(coeffs):
            constraints = []
            for i in range(num_segments):
                # Starting point of the segment
                constraints.append(np.polyval(coeffs[7*i:7*i+8][::-1], 0) - self.waypoints[i])
                # Ending point of the segment
                constraints.append(np.polyval(coeffs[7*i:7*i+8][::-1], self.times[i+1] - self.times[i]) - self.waypoints[i+1])
            return np.concatenate(constraints)

        result = minimize(self._polynomial_cost, initial_guess, constraints={'type': 'eq', 'fun': constraint_eq})
        if not result.success:
            print("Optimization failed:", result.message)
        return result.x.reshape(num_segments, 8)

    def _generate_trajectory_points(self):
        trajectory_points = []
        for i in range(len(self.coefficients)):
            coeffs = self.coefficients[i]
            T = np.arange(0, self.times[i+1] - self.times[i], self.dt)
            for t in T:
                position = np.polyval(coeffs[::-1], t)
                trajectory_points.append(position)
        return np.array(trajectory_points)

    def plot_trajectory(self):
        trajectory_points = self._generate_trajectory_points()
        plt.plot(trajectory_points, 'r-', label='Trajectory')
        plt.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='b', label='Waypoints')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Minimum Snap Trajectory')
        plt.legend()
        plt.show()

# Example usage
waypoints = [(0, 0), (2, 3), (4, 1), (6, 5)]
average_velocity = 1.0

planner = MinimumSnapTrajectory(waypoints, average_velocity)
planner.plot_trajectory()
