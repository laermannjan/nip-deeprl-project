import abc
import os
from collections import namedtuple
import gym
import datetime
import argparse

import tensorflow as tf
import tensorflow.contrib.layers as layers

from baselines import deepq
from baselines import logger

from configs import Configs

class Trainer:
    __metaclass__ = abc.ABCMeta
    def __init__(self, env_id, config_name, pickle_root, exp_name):
        self.env_id = env_id
        self.config_name = config_name
        self.pickle_root = pickle_root
        self.exp_name = exp_name

    @abc.abstractmethod
    def _train(self):
        pass

    def train(self):
        return self._train()

class SimpleTrainer(Trainer):
    def __init__(self, env_id, config_name, pickle_root, exp_name, is_solved_func=None, num_cpu=8):
        super().__init__(env_id, config_name, pickle_root, exp_name)
        self.num_cpu = num_cpu
        self.is_solved_func = is_solved_func

    def _train(self):
        env = gym.make(self.env_id)
        config = Configs[self.config_name]
        is_solved_func = self.is_solved_func

        act = deepq.learn(
            env,
            q_func=deepq.models.mlp(config.num_nodes),
            lr=config.learning_rate,
            max_timesteps=config.max_timesteps,
            buffer_size=config.replay_buffer_size,
            exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.final_p,
            train_freq=config.train_freq,
            batch_size=config.minibatch_size,
            print_freq=config.print_freq,
            checkpoint_freq=config.checkpoint_freq,
            learning_starts=config.learning_delay,
            gamma=config.gamma,
            target_network_update_freq=config.update_freq,
            callback=is_solved_func
        )
        exp_dir = '{}_{}'.format(env.spec.id, self.exp_name)
        pickle_dir = os.path.join(self.pickle_root, exp_dir)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        pickle_fname = '{}_{}.pkl'\
                       .format(env.spec.id, self.config_name)
        if config.print_freq is not None:
            logger.log("Saving model as {}".format(pickle_fname))
        act.save(os.path.join(pickle_dir, pickle_fname))


def is_solved_acrobot(lcl, glb, min_t_solved, mean_window, min_mean_reward):
    is_solved = lcl['t'] > min_t_solved and\
                sum(lcl['episode_rewards'][-(mean_window+1):-1]) / mean_window >= min_mean_reward
    return is_solved
