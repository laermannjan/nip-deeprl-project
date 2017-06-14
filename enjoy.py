import gym
import argparse
import os

from baselines import deepq
import tensorflow as tf

def enjoy(env_id, agent_fname):
    env = gym.make(env_id).env
    act = deepq.load(agent_fname)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='enjoy',
                                     description='Let a trained agent play an OpenAI Gym env.')
    parser.add_argument('env',
                        action='store',
                        choices=['Cartpole-v0', 'LunarLander-v2', 'Acrobot-v1'],
                        metavar='ENV',
                        type=str,
                        help='OpenAI Gym Id of the environment.')
    parser.add_argument('agent_fname',
                        action='store',
                        metavar='FNAME',
                        type=str,
                        help='''File name of the pickled agent. 
                        No safety checks if this is actually a pickled agent.''')

    args = parser.parse_args()
    if not os.path.isfile(args.agent_fname):
        parser.error('Your specified agent cannot be found.')

    enjoy(args.env, args.agent_fname)
