import gymnasium as gym
from uav_env import UAVEnv
import numpy as np
from plotting import *
import time
from skrl.envs.wrappers.torch import wrap_env


# load the models
import torch.nn as nn
import torch
# define models (stochastic and deterministic models) using mixins
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("Pendulum-v1")
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.vector.make(env_id, num_envs=4, asynchronous=False)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.9
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 2000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/Pendulum"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()

# evaluate the agent(s)
trainer.eval()

exit(0)
# wrap the environment

# Register the environment if you want to use gym.make()
gym.envs.registration.register(id='UAV-v0', entry_point='uav_env:UAVEnv')


# Create the environment
max_timesteps = 10
env = UAVEnv()
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'



exit(0)


n_episodes = 3

all_rewards = []
episode_lengths = []
all_actions = []

for episode in range(n_episodes):
    print("------------- begin episode", episode)
    state = env.reset()
    done = False
    total_reward = 0
    episode_length = 0
    episode_actions = []
    

    while not done:
        action = env.action_space.sample()  # Replace with actual RL agent's action
        print("selecting action", action)
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        episode_actions.append(action)

        # observation has vector [px,pz,vy, dp,dv] of size 9x1
        print("observations:", np.round(observation,1))
        print("reward:", np.round(reward,1))

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





