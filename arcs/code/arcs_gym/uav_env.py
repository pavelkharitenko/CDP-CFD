import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ar_controller import UAVSimulator  # Import your simulator class

class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()
        
        # Initialize your UAV simulator
        self.simulator = UAVSimulator()

        n_rotors = 4
        # Define the action space for rotor speeds in rps (continuous between -400 and 400)
        self.action_space = spaces.Box(
            low=np.full(n_rotors, -400.0,dtype=np.float64),   # Min rps for each rotor
            high=np.full(n_rotors, 400.0, dtype=np.float64),   # Max rps for each rotor
            dtype=np.float64,
            shape=(4,)
        )

        # min & max values of [px,pz,vy, dp_xyz, dv_xyz], a 9x1 array
        self.observation_space = spaces.Box(
            low=np.full(9, -5.0, dtype=np.float64),
            high=np.full(9, 5.0, dtype=np.float64),
            dtype=np.float64,
            shape=(9,)
        )

        print("###########", self.observation_space)

        
        # Define action and observation space
        # Example: Discrete action space (e.g., 4 directions) or continuous (e.g., velocity controls)
        
        # Observation space could be multidimensional (e.g., position, velocity)
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation"""
        initial_state = self.simulator.reset()
        info = {}
        return initial_state, info
    
    def step(self, action):
        """Take a step in the environment using the given action"""
        obs, reward, done, truncated, info = self.simulator.step(action)
        return obs, reward, done, truncated, info
    
    
    def close(self):
        """Clean up the environment (optional)"""
        self.simulator.close()
