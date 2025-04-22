
import numpy as np
import torch
import torch.nn.functional as F
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

        # Q function
        self.qf_1 = CriticQ(obs_dim + act_dim).to(self.device)
        self.qf_2 = CriticQ(obs_dim + act_dim).to(self.device)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.lr_vf)
        self.qf_1_optimizer = torch.optim.Adam(self.qf_1.parameters(), lr=self.lr_qf)
        self.qf_2_optimizer = torch.optim.Adam(self.qf_2.parameters(), lr=self.lr_qf)

        self.transition = list()
        self.total_steps = 0

        self.is_test = False





    def _init_hyperparams(self):
        self.gamma = 0.99
        self.tau = 5e-3
        self.memory_size = 5000
        self.batch_size = 1000
        self.initial_random_steps = 10000
        self.policy_update_freq = 2
        self.lr_alpha = 3e-4
        self.lr_actor = 3e-4
        self.lr_vf = 3e-4
        self.lr_qf = 3e-4

        

        self.seed = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state):
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(torch.FloatTensor(state).to(self.device))
            selected_action = selected_action[0].detach().cpu().numpy()
        
        self.transition = [state, selected_action]
        return selected_action



    def step(self, action):
        
        next_state, rew, terminated, truncated, _ = self.env.step(action)

        done = terminated or truncated

        if not self.is_test:
            self.transition +=[rew, next_state, done]
            self.memory.store(*self.transition)

        return next_state, rew, done


    def update_model(self):
        dev = self.device

        samples = self.memory.sample()
        state = to_tf(samples["obs"], dev)
        next_state = to_tf(samples["next_obs"], dev)
        action = to_tf(samples["acts"], dev)
        reward = to_tf(samples["rew"], dev)
        done = to_tf(samples["done"], dev)

        new_action, log_prob = self.actor(state)

        # train alpha
        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()


        alpha = self.log_alpha.exp()

        # Q function loss
        mask = 1 - done
        q_1_pred = self.qf_1(state, action)
        q_2_pred = self.qf_2(state, action)
        v_target = self.vf_target(next_state)
        q_target = reward + self.gamma * v_target * mask
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

    	# V function loss
        v_pred = self.vf(state)
        q_pred = torch.min(self.qf_1(state, new_action), self.qf_1(state, new_action))
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_step % self.policy_update_freq == 0:

            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update vf
            self._target_soft_update()
        
        else:
            actor_loss = torch.zeros(())

        # train Q function
        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        qf_loss = qf_1_loss + qf_2_loss

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.data, qf_loss.data, vf_loss.data, alpha_loss.data 








    def train(self):
        pass

    def test(self):
        pass

def to_tf(x, device):
    return torch.FloatTensor(x).to(device)