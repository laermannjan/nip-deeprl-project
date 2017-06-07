#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import random
from collections import deque

import gym
from gym import wrappers

import os
import logging.config
import yaml
import datetime

# Get rid of TF warnings concerning TF not being compiled from source.
# this might also hide other stuff (haven't checked yet)
# compiling from source supposed to give 3-8x performance boost depending on data size)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Setup
LOGGING_PATH = '../logging/'
LOG_LEVEL = logging.INFO
MONITORING_PATH = '../monitoring/'
MONITORING = False

ENV_ID = 'Acrobot-v1'
env = gym.make(ENV_ID)

# Hyperparams
HYPER = {
    'input_size' : env.observation_space.shape[0],
    'output_size' : env.action_space.n,
    'replay_memory' : 50000,
    'minibatch_size' : 10,
    'hidden_size' : 100,
    'max_episodes' : 50,
    'discount' : 0.9,
    'learning_rate' : 1e-1
}

def logger_setup(timestamp):
    log_file = '{}{}_{}.log'.format(
        LOGGING_PATH,
        ENV_ID,
        timestamp)

    formatter = logging.Formatter('[%(asctime)s] (%(levelname)s): %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)

    return logger


class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=HYPER['hidden_size'], l_rate=HYPER['learning_rate']):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            # First layer of weights
            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            # Second layer of Weights
            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            # Q prediction
            self._Qpred = tf.matmul(layer1, W2)

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})


def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predic(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # get target from target DQN (Q')
            Q[0, action] = reward + HYPER['discount'] * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack( [x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where
            # a = argmax(mainDQN(s'))
            Q[0, action] = reward + HYPER['discount'] * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN, env=env):
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    while True:
        # env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            return reward_sum

def random_baseline(env=env):
    # Set a baseline with an agent taken random actions only
    state = env.reset()
    reward_sum = 0
    while True:
        action = env.action_space.sample()
        state, reward, done, _ =  env.step(action)
        reward_sum += reward
        if done:
            return reward_sum

def compare_to_baseline(env, agent, n=200):
    logger.info('Benchmarking against random baseline...')
    if MONITORING:
        save_dir = '{}versus/{}'.format(MONITORING_PATH, agent)
        env = wrappers.Monitor(env, save_dir)

    baseline_reward = [random_baseline(env) for _ in range(n)]
    agent_reward  = [bot_play(agent, env) for _ in range(n)]

    logger.debug('Baseline performances: {}'.format(baseline_reward))
    logger.debug('Agent performances: {}'.format(agent_reward))

    baseline_reward = np.mean(baseline_reward)
    agent_reward = np.mean(agent_reward)

    logger.info('Avg. baseline performance: {}'.format(baseline_reward))
    logger.info('Avg. agent performance: {}'.format(agent_reward))

    return baseline_reward, agent_reward

def main():
    # store the previous observations in replay memory
    replay_buffer = deque()
    logger.debug('Initializing Replay Buffer as: {}'.format(replay_buffer))

    with tf.Session() as sess:
        logger.debug('Initializing main and target network...')
        mainDQN = DQN(sess, HYPER['input_size'], HYPER['output_size'], name="main")
        targetDQN = DQN(sess, HYPER['input_size'], HYPER['output_size'], name="target")
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(HYPER['max_episodes']):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                # env.render()
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                    logger.debug('Episode {}. Step {}. Random action chosen: {}'.format(episode, step_count, action))
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))
                    logger.debug('Episode {}. Step {}. Greedy action chosen: {}'.format(episode, step_count, action))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action) # last return gives 'observations', might be useful

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                logger.debug('Adding to replay buffer: {} -> {}. Reward: {}, Done: {}.'.format(state, next_state, reward, done))
                if len(replay_buffer) > HYPER['replay_memory']:
                      replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:   # Good enough. Let's move on
                    break

            # print("Episode: {} steps: {}".format(episode, step_count))
            if step_count > 10000:
                pass
                # break

            if episode % 10 == 1: # train every 10 episode
                # Get a random batch of experiences
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, HYPER['minibatch_size'])
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)

                logger.info('Episode {} finished after {} steps with a loss of: {}'.format(episode, step_count, loss))
                # copy q_net -> target_net
                sess.run(copy_ops)
            else:
                logger.info('Episode {} finished after {} steps.'.format(episode, step_count))

        compare_to_baseline(env, mainDQN)


        # See our trained bot in action
        # env2 = wrappers.Monitor(env, 'gym-results', force=True)

        # for i in range(200):
            # bot_play(mainDQN, env=env2)

        # env2.close()
        # gym.upload("gym-results", api_key="sk_VT2wPcSSOylnlPORltmQ")

if __name__ == "__main__":
    timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
    logger = logger_setup(timestamp)
    logger.info('Initializing experiment with these Hyperparams: {}'.format(HYPER))
    if MONITORING:
        monitor_dir = '{}{}/training'.format(MONITORING_PATH, timestamp)
        env = wrappers.Monitor(env, monitor_dir)
        logger.info('Monitor output saved here: {}'.format(monitor_dir))
    main()
    # if PLAY:
    #     monitor_dir = '{}{}/playing'.format(MONITORING_PATH, timestamp)
    #     env2 = wrappers.Monitor(env, monitor_dir)
    #     for _ in range(200):
    #         pass
