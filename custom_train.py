import gym
import itertools
import numpy as np
import datetime
import argparse

import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
from baselines.deepq.simple import ActWrapper

from collections import namedtuple
import os
import tempfile

from configs import Configs


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions.

    It relies on hyperparameter configuration being defined with name *config*.

    Parameters
    ----------
    inpt : array-like
        Current observations of the environment.

    num_actions : integer
        Number of discrete actions available to the agent.

    scope : str
        TensorFlow variable scope

    reuse : boolean, Default: False
        Determines if tf session is reusable.
    """


    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        # for n_nodes, activation_fn in zip(config.num_nodes, config.act_fns):
            # out = layers.fully_connected(out, num_outputs=n_nodes, activation_fn=activation_fn)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def train(env, config, num_cpu=8):
    with U.make_session(num_cpu):
        def make_obs_ph(name):
            return U.BatchInput(env.observation_space.shape, name=name)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=config.optimizer(learning_rate=config.learning_rate),
        )

        exploration_schedule = config.exploration_schedule(
            schedule_timesteps=config.schedule_timesteps,
            initial_p=config.initial_p,
            final_p=config.final_p)

        replay_buffer = ReplayBuffer(config.replay_buffer_size)
        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        episode_lengths = [0]
        obs = env.reset()
        saved_mean_reward = None
        with tempfile.TemporaryDirectory() as td:
            model_saved = False
            model_file  = os.path.join(td, 'model')
            for t in range(config.max_timesteps):
                # Take action and update exploration to the newest value
                action = act(obs[None], update_eps=exploration_schedule.value(t))[0]
                new_obs, rew, done, _ = env.step(action)
                rew = rew if not done else config.done_reward
                done = done if episode_lengths[-1] != config.max_timesteps_ep else True

                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                episode_lengths[-1] += 1

                mean_reward = np.mean(episode_rewards[-config.mean_window:])
                mean_ep_len = np.mean(episode_lengths[-config.mean_window:])


                is_solved = t > config.min_t_solved and mean_reward >= config.min_mean_reward
                if not is_solved:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if t > config.learning_delay and t % config.train_freq == 0:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(config.minibatch_size)
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # Update target network periodically.
                    if t % config.update_freq == 0:
                        update_target()

                if config.print_freq is not None and done and len(episode_rewards) % config.print_freq == 0:
                    logger.record_tabular("total steps", t)
                    logger.record_tabular("episodes", len(episode_rewards) - 1)
                    logger.record_tabular("mean episode length", round(mean_ep_len, 0))
                    logger.record_tabular("mean episode reward", round(mean_reward, 1))
                    logger.record_tabular("% time spent exploring", int(100 * exploration_schedule.value(t)))
                    logger.dump_tabular()

                if config.checkpoint_freq is not None and t > config.learning_delay and t % config.checkpoint_freq == 0:
                    if saved_mean_reward is None or mean_reward > saved_mean_reward:
                        if config.print_freq is not None:
                            logger.log('Saving model due to mean reward increase: {} -> {}'.format(saved_mean_reward, mean_reward))
                        U.save_state(model_file)
                        saved_mean_reward = mean_reward
                        model_saved = True

                if is_solved:
                    break

                if done:
                    obs = env.reset()
                    episode_rewards.append(0)
                    episode_lengths.append(0)

            if model_saved:
                U.load_state(model_file)
                act_params = {
                    'make_obs_ph' : make_obs_ph,
                    'num_actions' : env.action_space.n,
                    'q_func' : model,
                }

                pickle_dir = 'trained_agents'
                if not os.path.exists(pickle_dir):
                    os.makedirs(pickle_dir)
                pickle_fname = '{}_model_{}_custom.pkl'.format(env.spec.id,
                                                        datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'))
                if config.print_freq is not None:
                    logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
                    logger.log("Saving model as {}".format(pickle_fname))
                ActWrapper(act, act_params).save(os.path.join(pickle_dir, pickle_fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='custom_train',
                                     description='Train agent to play an OpenAI gym env.')
    parser.add_argument('env',
                        action='store',
                        choices=['Cartpole-v0', 'LunarLander-v2', 'Acrobot-v1'],
                        metavar='ENV',
                        type=str,
                        help='OpenAI Gym Id of the environment.')
    parser.add_argument('config',
                        action='store',
                        choices=Configs.keys(),
                        metavar='CONFIG',
                        type=str,
                        help='Name of config from configs.py.')
    parser.add_argument('--dir',
                        action='store',
                        type=str,
                        dest='pickle_dir',
                        default='trained_agents',
                        help='Directory to save trained agent.')
    parser.add_argument('--cores',
                        action='store',
                        type=int,
                        dest='num_cpu',
                        default=8,
                        help='Number of cpu cores to be used for training.')
    args = parser.parse_args()

    env = gym.make(args.env).env
    config = Configs[args.config]
    num_cpu = args.num_cpu
    train(env, config, num_cpu)