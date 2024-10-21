import gymnasium as gym
from uav_env import UAVEnv

# Register the environment if you want to use gym.make()
#gym.envs.registration.register(id='UAV-v0', entry_point='uav_env:UAVEnv')

# Create the environment
env = UAVEnv()

# Example interaction loop
state = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with actual RL agent's action
    state, reward, done, truncated, info = env.step(action)
    
    print(state)

print("done!")
env.close()
