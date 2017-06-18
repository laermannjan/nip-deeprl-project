import argparse
import os
import gym

from baselines import deepq
import tensorflow as tf

class SimpleEvaluator:
    def __init__(self, env_id, agent_fname):
        self.env_id = env_id
        self.agent_fname = agent_fname


    def eval(self):
        env = gym.make(self.env_id)
        act = deepq.load(self.agent_fname)

        while True:
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                env.render()
                obs, rew, done, _ = env.step(act(obs[None])[0])
                episode_rew += rew
            print("Episode reward", episode_rew)
