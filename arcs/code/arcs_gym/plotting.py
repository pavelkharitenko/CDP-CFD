import matplotlib.pyplot as plt
import numpy as np



# Plot total rewards per episode
def plot_rewards(all_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, label="Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot episode lengths (number of steps per episode)
def plot_episode_lengths(episode_lengths):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths, label="Episode Lengths", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.title("Episode Lengths")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the distribution of actions taken
def plot_action_distribution(all_actions, action_space_size):
    # Flatten list of actions
    actions = np.concatenate(all_actions)
    
    plt.figure(figsize=(10, 5))
    plt.hist(actions, bins=np.arange(action_space_size + 1) - 0.5, rwidth=0.8, color="green")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Action Distribution")
    plt.xticks(np.arange(action_space_size))
    plt.grid(True)
    plt.show()

# Plot cumulative reward over time (sum of rewards)
def plot_cumulative_rewards(all_rewards):
    cumulative_rewards = np.cumsum(all_rewards)
    
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_rewards, label="Cumulative Reward", color="purple")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_metrics(all_rewards, episode_lengths, all_actions, action_space_size):
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid of plots
    
    # Plot 1: Total rewards per episode
    axs[0, 0].plot(all_rewards, label="Rewards", color="blue")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].set_title("Total Reward per Episode")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Episode lengths (steps per episode)
    axs[0, 1].plot(episode_lengths, label="Episode Lengths", color="orange")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Number of Steps")
    axs[0, 1].set_title("Episode Lengths")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Action distribution (histogram)
    actions = np.concatenate(all_actions)  # Flatten the list of actions
    axs[1, 0].hist(actions, bins=np.arange(action_space_size + 1) - 0.5, rwidth=0.8, color="green")
    axs[1, 0].set_xlabel("Action")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].set_title("Action Distribution")
    axs[1, 0].set_xticks(np.arange(action_space_size))
    axs[1, 0].grid(True)

    # Plot 4: Cumulative rewards
    cumulative_rewards = np.cumsum(all_rewards)
    axs[1, 1].plot(cumulative_rewards, label="Cumulative Reward", color="purple")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Cumulative Reward")
    axs[1, 1].set_title("Cumulative Reward Over Time")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

