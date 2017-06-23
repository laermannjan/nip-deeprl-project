import argparse
import os
import gym

from baselines import deepq
import tensorflow as tf

def eval(game_name, agent_name):
    env = gym.make(game_name)
    act = deepq.load(agent_name)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
