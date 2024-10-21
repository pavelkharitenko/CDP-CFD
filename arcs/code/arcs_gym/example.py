import gymnasium as gym
from uav_env import UAVEnv
from plotting import *

# Register the environment if you want to use gym.make()
#gym.envs.registration.register(id='UAV-v0', entry_point='uav_env:UAVEnv')

# Create the environment

n_episodes = 5

all_rewards = []
episode_lengths = []
all_actions = []

for episode in range(n_episodes):
    print("begin episode", episode)
    env = UAVEnv()
    state = env.reset()
    done = False
    total_reward = 0
    episode_length = 0
    episode_actions = []
    

    while not done:
        action = env.action_space.sample()  # Replace with actual RL agent's action
        print("selecting action", action)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        episode_actions.append(action)

        
        imu1 = state[0][0]
        imu2 = state[0][1]
        current_time = state[1]
        print(current_time)
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
plot_all_metrics(all_rewards, episode_lengths, all_actions, env.action_space.n)





