import torch
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
import time
from gymnasium import spaces

from networks import FeedForwardNN

class PPO:
    def __init__(self, env, env_name):

        self._init_hyper_parameters()

        env = self.adjust_env_spaces(env_name, env)

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

        self.plotter = {
            "avg_ep_lens": [],
            "avg_ep_rews": [],
            "avg_actor_loss": [],
            "avg_critic_loss": [],
            "timesteps": [],
        }

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
            "critic_losses": [],

		}

    def _init_hyper_parameters(self):
        self.timesteps_per_batch = 1000
        self.max_timesteps_per_episode = 1000
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


        # collect set of trajectories D with policy pi_theta
        t = 0 # number of timesteps for this batch
        while t < self.timesteps_per_batch:
            ep_rews = []

            obs, _ = self.env.reset()
            
            if isinstance(obs, dict):
                obs = obs["observation"]
            
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                
                # collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                if isinstance(obs, dict): # handle pandagym
                    obs = obs["observation"]

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

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        

        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs # return V_phi_k and pi_new

    def learn(self, total_timesteps):
        t_so_far = 0
        i_so_far = 0

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
                #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.logger['actor_losses'].append(actor_loss.detach())
                
                # fit V_t estimate on regression loss: phi_new = ... ( V_phi - R_t )^2 via gradient descent
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
                self.logger["critic_losses"].append(critic_loss.detach())




            t_so_far += np.sum(batch_lens)

			# Track number of iterations/rollouts 
            i_so_far += 1

			# Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            self._log_summary()
        
        self._plot_summary()
    
    def _log_summary(self):
        """
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_losses']])


		# Round decimal places for more aesthetic logging messages
        avg_ep_lens = round(avg_ep_lens, 2)
        avg_ep_rews = round(avg_ep_rews, 2)
        avg_actor_loss = round(avg_actor_loss, 5)
        avg_critic_loss = round(avg_critic_loss, 5)


		# Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)

        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        self.plotter["avg_ep_lens"].append(avg_ep_lens)
        self.plotter["avg_ep_rews"].append(avg_ep_rews)
        self.plotter["avg_actor_loss"].append(avg_actor_loss)
        self.plotter["avg_critic_loss"].append(avg_critic_loss)
        self.plotter["timesteps"].append(t_so_far)


		# Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []

    def _plot_summary(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        metrics = ['avg_ep_lens', 'avg_ep_rews', 'avg_actor_loss', 'avg_critic_loss']
        titles = ['Episode Lengths', 'Episode Rewards', 'Actor Loss', 'Critic Loss']
        
        for ax, metric, title in zip(axs.flatten(), metrics, titles):
            ax.plot(self.plotter['timesteps'], self.plotter[metric])
            ax.set_title(title)
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        

    def adjust_env_spaces(self, env_name, env, end_effector_dims=3):
        # Check if it's a Panda-Gym environment (always continuous)
        if "Panda" in env_name:
            print("Panda-Gym environment detected. Using Box spaces.")
            
            # Modify observation space if needed (e.g., end-effector control)
            obs_shape = env.observation_space.shape
            #print("obs shape", obs_shape)
            new_obs_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            )
            
            # Modify action space for end-effector control (e.g., 3D position)
            new_action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(end_effector_dims,), dtype=np.float32
            )

            #print("new obs shape", new_obs_space.shape[0])

            
            env.observation_space = new_obs_space
            env.action_space = new_action_space
        
        # Handle standard Gym environments (Discrete or Box)
        else:
            print("Standard Gym environment detected.")
            
            # If action space is Discrete, convert to Box (one-hot-like)
            if isinstance(env.action_space, spaces.Discrete):
                print("Discrete action space detected. Converting to Box.")
                n_actions = env.action_space.n
                env.action_space = spaces.Box(
                    low=0, high=1, shape=(n_actions,), dtype=np.float32
                )
            
            # If observation space is Discrete, convert to Box (one-hot)
            if isinstance(env.observation_space, spaces.Discrete):
                print("Discrete observation space detected. Converting to Box.")
                n_obs = env.observation_space.n
                env.observation_space = spaces.Box(
                    low=0, high=1, shape=(n_obs,), dtype=np.float32
                )
        
        return env


        



