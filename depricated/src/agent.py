import numpy as np
from collections import deque
import random

import tensorflow as tf

import dqn
from utils import EpsilonStrategy as e_strat
from utils import train_agent
from utils import compare_agents


class VanillaAgent(object):
    """Vanilla Agent. No significant changes from reference implementation."""

    def __init__(self, action_space, config=None):
        self.action_space = action_space
        if config is None:
            config = {
                'replay_buffer_size' : 50000,
                'minibatch_size' : 10,
                'hidden_size' : 10,
                'max_episodes' : 50,
                'max_steps' : 10000,
                'discount' : 0.9,
                'learning_rate' : 1e-1,
                'train_freq' : 10,
                'train_length' :  50,
                'eps_strategy' : e_strat.GREEDY
            }
        self.config = config
        logger.info('VanillaAgent initialized with config: {}'.format(config))
        self.replay_buffer = deque()

    def _get_copy_var_ops(self, dest_scope_name='target', src_scope_name='main'):
        """Copy variables src_scope to dest_scope"""
        op_holder = []

        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder

    def _update_epsilon(self, episode):
        if self.config['eps_strategy'] == e_strat.GREEDY:
            return 1. / ((episode / 100) + 1) #TODO: these vals need to be reviewed!!


    def act(self, state, eps, sess):
        if np.random.random() > eps:
            action = np.argmax(self.main_dqn.predict(state))
            logger.debug('Network chose action: {}'.format(action))
        else:
            action =  self.action_space.sample()
            logger.debug('Action was chosen at random: {}'.format(action))
        return action

    def play(self, env, sess):
        state = env.reset()
        reward_sum = 0
        while True:
            action = self.act(state, 0, sess)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                return reward_sum

    def learn(self, env, sess):
        self.env = env
        input_size = self.env.observation_space.shape[0]
        output_size = self.env.action_space.n

        logger.info('Initializing main and target networks...')
        self.main_dqn = dqn.TwoLayerDQN(sess, input_size, self.config['hidden_size'],
                                output_size, self.config['learning_rate'], name='main')
        self.target_dqn = dqn.TwoLayerDQN(sess, input_size, self.config['hidden_size'],
                                  output_size, self.config['learning_rate'], name='target')
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_network
        copy_ops = self._get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
        sess.run(copy_ops)

        for episode in range(self.config['max_episodes']):
            eps = self._update_epsilon(episode)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                action = self.act(state, eps, sess)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                logger.debug('Added to replay buffer: {}'\
                             .format((state, action, reward, next_state, done)))
                if len(self.replay_buffer) > self.config['replay_buffer_size']:
                    self.replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > self.config['max_steps']:
                    break

            if not episode % self.config['train_freq']:
                for _ in range(self.config['train_length']):
                    minibatch = random.sample(self.replay_buffer,
                                              self.config['minibatch_size'])
                    loss, _ = dqn.ddqn_replay_train(self.main_dqn,
                                                    self.target_dqn,
                                                    minibatch,
                                                    self.config['discount'])
                sess.run(copy_ops)
                logger.info('Episode {} finished after {} steps with a loss of: {}'\
                            .format(episode, step_count, loss))
            else:
                logger.info('Episode {} finished after {} steps.'.format(episode, step_count))


if __name__ == '__main__':
    import os
    import logging
    import gym
    from utils import logger_setup
    # Get rid of TF warnings concerning TF not being compiled from source.
    # this might also hide other stuff (haven't checked yet)
    # compiling from source supposed to give 3-8x performance boost depending on data size)
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    env = gym.make('Acrobot-v1').env
    logger = logger_setup(env.spec.id)
    # logger = logger_setup(env.spec.id,log_level=logging.DEBUG)

    # custom agent config
    config = {
        'replay_buffer_size' : 100000,
        'minibatch_size' : 32,
        'hidden_size' : 100,
        'max_episodes' : 500,
        'max_steps' : 10000,
        'discount' : 0.99,
        'learning_rate' : 1e-2,
        'train_freq' : 5,
        'train_length' :  500,
        'eps_strategy' : e_strat.GREEDY
     }


    agent = VanillaAgent(env.action_space, config=config)
    with tf.Session() as sess:
        train_agent(agent, env, sess, logger=logger)
        #train_agent(agent, env, monitor_root_dir='../monitoring/')

        from random_agent import RandomAgent
        rand_agent = RandomAgent(env.action_space)
        compare_agents(agent, rand_agent, env, sess, logger=logger)
