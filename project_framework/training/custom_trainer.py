import os
import gym
import numpy as np
import datetime
import argparse
import itertools
import tempfile
from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as U
import baselines.common.misc_util as mu
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
from baselines.deepq.simple import ActWrapper

from configs import Configs

from project_framework.training import Trainer
from project_framework.wrappers import Monitor

class CustomTrainer(Trainer):
    def __init__(self, env_id, config_name, pickle_root, exp_name, is_solved_func=None, num_cpu=8):
        super().__init__(env_id, config_name, pickle_root, exp_name)
        self.num_cpu = num_cpu
        self.is_solved_func = is_solved_func

    def _train(self):
        config = Configs[self.config_name]
        env = gym.make(self.env_id)
        env._max_episode_steps = config.max_timesteps_ep
        env = Monitor(env, directory=self.exp_name, video_callable=False, force=True,
                      write_upon_reset=True, # as this writes JSON files, might be really slow!
                      uid=None, #uses os.getpid() to create file suffix
                      mode='training')
        env.set_augmented_reward(config.done_reward)
        model = deepq.models.mlp(config.num_nodes)

        with U.make_session(self.num_cpu):
            def make_obs_ph(name):
                return U.BatchInput(env.observation_space.shape, name=name)

            # Create all the functions necessary to train the model
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=make_obs_ph,
                q_func=model,
                num_actions=env.action_space.n,
                optimizer=config.optimizer(learning_rate=config.learning_rate),
            )
            env.register_graph(act, train, update_target)

            exploration_schedule = config.exploration_schedule(
                schedule_timesteps=config.schedule_timesteps,
                initial_p=config.initial_p,
                final_p=config.final_p)

            replay_buffer = ReplayBuffer(config.replay_buffer_size)
            # Initialize the parameters and copy them to the target network.
            U.initialize()
            update_target()

            obs = env.reset()
            saved_mean_reward = None
            with tempfile.TemporaryDirectory() as td:
                model_saved = False
                model_file = os.path.join(td, 'model')
                for t in range(config.max_timesteps):
                    # Take action and update exploration to the newest value
                    action = env.act(obs[None], update_eps=exploration_schedule.value(t))[0]
                    new_obs, rew, done, info = env.step(action)
                    # Store transition in the replay buffer.
                    replay_buffer.add(obs, action, rew, new_obs, float(done))
                    obs = new_obs

                    if done:
                        obs = env.reset()
                        # Log to console
                        if config.print_freq and\
                            len(env.get_mean_episode_rewards()) % config.print_freq == 0:
                            self.log_timestamp()
                        # Create Checkpoint
                        if config.checkpoint_freq and\
                           t > config.learning_delay and t % config.checkpoint_freq == 0:
                            if saved_mean_reward is None or env.get_current_mean_reward() > saved_mean_reward:
                                if config.print_freq:
                                    logger.log('Saving model due to mean reward increase: {} -> {}'\
                                               .format(saved_mean_reward, env.get_current_mean_reward()))
                                U.save_state(model_file)
                                saved_mean_reward = env.get_current_mean_reward()
                                model_saved = True

                        if not self.solved(t, config):
                            # Minimize the error in Bellman's equation on a batch sampled from replay buffer
                            if t > config.learning_delay and t % config.train_freq == 0:
                                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(config.minibatch_size)
                                env.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                            # Update target network periodically.
                            if t % config.update_freq == 0:
                                env.update_target()
                        else:
                            break

                if model_saved:
                    U.load_state(model_file)
                    act_params = {
                        'make_obs_ph' : make_obs_ph,
                        'num_actions' : env.action_space.n,
                        'q_func' : model,
                    }

                    exp_dir = '{}_{}'.format(env.spec.id, self.exp_name)
                    pickle_dir = os.path.join(self.pickle_root, exp_dir)
                    if not os.path.exists(pickle_dir):
                        os.makedirs(pickle_dir)
                        pickle_fname = '{}_{}.pkl'\
                                       .format(env.spec.id, self.config_name)
                    if config.print_freq is not None:
                        logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
                        logger.log("Saving model as {}".format(pickle_fname))
                    ActWrapper(act, act_params).save(os.path.join(pickle_dir, pickle_fname))

    def _log_timestamp(self, env):
        logger.record_tabular("total steps", env.get_total_steps())
        logger.record_tabular("total episodes", len(env.get_episode_lengths()))
        logger.record_tabular("mean episode length", env.get_mean_episode_lengths()[-1])
        logger.record_tabular("mean episode reward", env.get_current_mean_reward())
        logger.record_tabular("% time spent exploring",
                              int(100 * exploration_schedule.value(t)))
        logger.dump_tabular()

    def solved(self, t, config):
         return t > config.min_t_solved and env.get_current_mean_reward() >= config.min_mean_reward
