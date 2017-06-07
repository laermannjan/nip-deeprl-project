import os
from collections import namedtuple
import gym

import tensorflow as tf
import tensorflow.contrib.layers as layers

from baselines import deepq
from baselines import logger

from configs import Configs


def is_solved_cartpole(lcl, glb, min_t_solved, mean_window, min_mean_reward):
    is_solved = lcl['t'] > min_t_solved and sum(lcl['episode_rewards'][-(mean_window+1):-1]) / mean_window >= min_mean_reward
    return is_solved

def is_solved_func(lcl, glb):
    return is_solved_cartpole(lcl, glb, config.min_t_solved, config.mean_window, config.min_mean_reward)


if __name__ == '__main__':
    config = Configs['cartpole_basic']

    env = gym.make(config.env)
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

    pickle_dir = 'trained_agents'
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    pickle_fname = 'cartpole_model.pkl' # TODO: Need to add some sort of id, maybe timestamp
    logger.log("Saving model as {}".format(pickle_fname))
    act.save(os.path.join(pickle_dir,pickle_fname))
