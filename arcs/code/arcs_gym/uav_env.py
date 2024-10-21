import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ar_controller import UAVSimulator  # Import your simulator class

class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()
        
        # Initialize your UAV simulator
        self.simulator = UAVSimulator()
        
        # Define action and observation space
        # Example: Discrete action space (e.g., 4 directions) or continuous (e.g., velocity controls)
        self.action_space = spaces.Discrete(4)  # Example: 4 possible actions (e.g., up, down, left, right)
        # Observation space could be multidimensional (e.g., position, velocity)
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
    def reset(self):
        """Reset the environment and return the initial observation"""
        initial_state = self.simulator.reset()
        info = {}
        return initial_state, info
    
    def step(self, action):
        """Take a step in the environment using the given action"""
        state, reward, done, truncated, info = self.simulator.step(action)
        return state, reward, done, truncated, info
    
    
    
    def close(self):
        """Clean up the environment (optional)"""
        pass
