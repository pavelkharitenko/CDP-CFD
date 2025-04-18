import gymnasium as gym
import sys, torch
from ppo import PPO
from networks import FeedForwardNN
import panda_gym


def test_rollout(env, policy, render):
    while True:
        obs, _ = env.reset()
        done = False
        t = 0
        ep_len = 0
        ep_ret = 0

        while not done:
            t += 1
            if render:
                env.render()

            if isinstance(obs, dict):
                obs = obs["observation"]
            action = policy(obs).detach().numpy()
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated

            if terminated:
                print("success!")
        
        ep_len += t

        yield ep_len, ep_ret


def test(env_name, policy, render=False):

    print("Testing")
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    for ep_num, (ep_len, ep_ret) in enumerate(test_rollout(env, policy, render)):
        ep_len = str(round(ep_len, 3))
        ep_ret = str(round(ep_ret, 3))

		# Print logging statements
        print(flush=True)
        print(f"-------------------- Eval. Episode #{ep_num} --------------------", flush=True)
        print(f"Episodic Length: {ep_len}", flush=True)
        print(f"Episodic Return: {ep_ret}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)
        

def train(env_name, hyperparameters=None, actor_model=None, critic_model=None, time_steps=5000):

    env = gym.make(env_name)
    

    print("Training...")

    model = PPO(env, env_name)
    model.learn(time_steps)

    return model.actor



def main():

    #env_name = 'Pendulum-v1'
    #env_name = 'LunarLanderContinuous-v2'
    #env_name = 'BipedalWalker-v3'
    env_name = "PandaReachDense-v3"


    actor = train(env_name, time_steps=1250000)

    test(env_name, actor, render=True)



main()