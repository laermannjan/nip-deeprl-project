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
    def __init__(self, env_id, config_name, root_dir, exp_name, videos_enabled=False, is_solved_func=None, num_cpu=8):
        super().__init__(env_id, config_name, root_dir, exp_name)
        self.num_cpu = num_cpu
        self.is_solved_func = is_solved_func
        self.videos_enabled = videos_enabled

    def _train(self):
        config = Configs[self.config_name]
        env = gym.make(self.env_id)
        env._max_episode_steps = config.max_timesteps_ep
        env = Monitor(env, directory=self.root_dir, video_callable=self.videos_enabled, force=False,
                      resume=True,
                      write_upon_reset=False, # as this writes JSON files, might be really slow!
                      uid=self.exp_name, 
                      mode='training')
        if config.done_reward is not None:
            env.set_augmented_reward(config.done_reward)
        env.config_name = self.config_name
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
            best_mean_reward = None
            with tempfile.TemporaryDirectory() as td:
                best_model_saved = False
                final_model_saved = False
                best_model_file = os.path.join(td, 'best_model')
                final_model_file = os.path.join(td, 'final_model')
                for t in range(config.max_timesteps):
                # for t in range(int(1e5)):
                    # Take action and update exploration to the newest value
                    eps = exploration_schedule.value(t)
                    action = act(obs[None], update_eps=eps)[0]
                    env._record_exploration(eps)
                    new_obs, rew, done, info = env.step(action)
                    # Store transition in the replay buffer.
                    replay_buffer.add(obs, action, rew, new_obs, float(done))
                    obs = new_obs


                    # Create Checkpoint
                    if config.checkpoint_freq and\
                       t > config.learning_delay and t % config.checkpoint_freq == 0:
                        if best_mean_reward is None or env.get_current_mean_reward() > best_mean_reward:
                            if config.print_freq:
                                logger.log('Saving model due to mean reward increase: {} -> {}'\
                                           .format(best_mean_reward, env.get_current_mean_reward()))
                            U.save_state(best_model_file)
                            best_mean_reward = env.get_current_mean_reward()
                            best_model_saved = True

                    if done:
                        obs = env.reset()
                        # Log to console
                        if config.print_freq and\
                            len(env.get_mean_episode_rewards()) % config.print_freq == 0:
                            self._log_timestamp(env, eps)

                        if not self.solved(env, t, config):
                            # Minimize the error in Bellman's equation on a batch sampled from replay buffer
                            if t > config.learning_delay and t % config.train_freq == 0:
                                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(config.minibatch_size)
                                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                                env._record_errors(td_errors)
                            # Update target network periodically.
                            if t % config.update_freq == 0:
                                update_target()
                        else:
                            break

                if t > config.learning_delay:
                    U.save_state(final_model_file)
                    final_model_saved = True
                act_params = {
                    'make_obs_ph' : make_obs_ph,
                    'num_actions' : env.action_space.n,
                    'q_func' : model,
                }
                # if final_model_saved:
                #     U.load_state(final_model_file)
                #     self.pickle_agent(act, act_params, 'final')
                # if best_model_saved:
                #     U.load_state(best_model_file)
                #     pickle_name = 'highest_reward_{}'.format(best_mean_reward)
                #     self.pickle_agent(act, act_params, pickle_name)

    def pickle_agent(self, act, act_params, name):
        pickle_dir = os.path.join(self.root_dir, 'saved_agents')
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        fname = 'LOG.agent.{}.pkl'.format(name)
        logger.log('Saving agent as {}.'.format(fname))
        ActWrapper(act, act_params).save(os.path.join(pickle_dir, fname))

    def _log_timestamp(self, env, eps):
        logger.record_tabular("total steps", env.get_total_steps())
        logger.record_tabular("total episodes", len(env.get_episode_lengths()))
        logger.record_tabular("mean episode length", env.get_mean_episode_lengths()[-1])
        logger.record_tabular("mean episode reward", env.get_current_mean_reward())
        logger.record_tabular("% time spent exploring", int(100 * eps))
        logger.dump_tabular()

    def solved(self, env, t, config):
         return t > config.min_t_solved and env.get_current_mean_reward() >= config.min_mean_reward
