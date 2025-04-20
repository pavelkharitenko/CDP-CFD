
import numpy as np
import torch

from .networks import Actor, CriticQ, CriticV

class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size, batch_size=32):
        
        self.buffer["obs"] = np.zeros([size, obs_dim], dtype=np.float32)
        self.buffer["acts"] = np.zeros([size, act_dim], dtype=np.float32)
        self.buffer["rews"] = np.zeros([size], dtype=np.float32)
        self.buffer["next_obs"] = np.zeros([size, obs_dim], dtype=np.float32)
        self.buffer["done"] = np.zeros([size], dtype=np.float32)
        self.buffer["done"] = np.zeros([size, obs_dim], dtype=np.float32)

        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0,0




    def store(self, state, action, reward, next_state, done):
        # store (s,a,r,s') tuples

        self.buffer["obs"][self.ptr] = state
        self.buffer["acts"][self.ptr] = action
        self.buffer["rews"][self.ptr] = reward
        self.buffer["next_obs"][self.ptr] = next_state
        self.buffer["done"][self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size # update circular pointer
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        if self.size < self.batch_size:
            raise ValueError("Not enough samples in ReplayBuffer")
        
        indices = np.random.choice(self.size, self.batch_size, replace=False)

        return {key: self.buffer[key][indices] for key in self.buffer}

    def __len__(self):
        return self.size



class SACAgent:
    def __init__(self, env, env_name):
        self._init_hyperparams()

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.env = env
        self.memory = ReplayBuffer(obs_dim, act_dim, self.memory_size, self.batch_size)

        # entropy alpha
        self.target_entropy = -np.prod((act_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)

        # actor
        self.actor = Actor(obs_dim, act_dim).to(self.device)

        # value function
        self.vf = CriticV(obs_dim).to(self.device)
        self.vf_target = CriticV(obs_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())




    def _init_hyperparams(self):
        self.gamma = 0.99
        self.tau = 5e-3
        self.memory_size = 5000
        self.batch_size = 1000
        self.initial_random_steps = 10000
        self.policy_update_freq = 2
        self.lr_alpha = 3e-4

        

        self.seed = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self):
        pass

    def update_model(self):
        pass

    def step(self):
        pass


    def train(self):
        pass

    def test(self):
        pass