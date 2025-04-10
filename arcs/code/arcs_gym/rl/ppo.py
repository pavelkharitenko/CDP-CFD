import torch
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np

from networks import FeedForwardNN

class PPO:
    def __init__(self, env):

        self._init_hyper_parameters()

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # 1 init networks and params theta
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyper_parameters(self):
        self.timesteps_per_batch = 1000
        self.max_timesteps_per_episode = 200
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 5e-4
        self.max_grad_norm = 0.5
   
    def get_action(self, obs):
        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        # compute rewards to go R_t
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float) # convert to tensor
        return batch_rtgs

    def rollout(self):
        # batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # doing rollout: 
        print("doing rollout...")

        # collect set of trajectories D with policy pi_theta
        t = 0 # number of timesteps for this batch
        while t < self.timesteps_per_batch:
            ep_rews = []

            obs, _ = self.env.reset()
            
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                
                # collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                # record in batch
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                done = terminated or truncated

                if done:
                    break

            # record episode length and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        

        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs # return V_phi_k and pi_new

    def learn(self, total_timesteps):
        t_so_far = 0

        # for k = 1,2,... do
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            V, _ = self.evaluate(batch_obs, batch_acts)

            # compute advantage estimates A_t from current V_t estimate
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # update policy:
                
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios*A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k

                # theta_new = argmax_theta = 1/len(D) * sum[t~D] sum[t=0,T] min(pi_new/pi_theta * A_, clip(eps, A_pi_theta) via gradient ascent
                actor_loss = (-torch.min(surr1, surr2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                
                # fit V_t estimate on regression loss: phi_new = ... ( V_phi - R_t )^2 via gradient descent
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.critic_optim.step()


            t_so_far += np.sum(batch_lens)


import gymnasium as gym
env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)