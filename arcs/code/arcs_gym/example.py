import gymnasium as gym
from uav_env import UAVEnv
import numpy as np
from plotting import *

# Register the environment if you want to use gym.make()
#gym.envs.registration.register(id='UAV-v0', entry_point='uav_env:UAVEnv')

# Create the environment
env = UAVEnv()
n_episodes = 1000

all_rewards, episode_lengths, all_actions = [], [], []

for episode in range(n_episodes):
    state = env.reset()
    done = False
    total_reward, episode_length = 0, 0
    episode_actions = []

    while not done:
        action = env.action_space.sample()  # Replace with actual RL agent's action
        
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        episode_actions.append(action)

        current_time = info
        print(np.round(current_time,2))
        # End the episode if done or truncated
        if done or truncated:
            break
    
    # Store statistics for this episode
    all_rewards.append(total_reward)
    episode_lengths.append(episode_length)
    all_actions.append(episode_actions)

env.close()
print("done!")

# evaluate
plot_all_metrics(all_rewards, episode_lengths, all_actions, 4)





