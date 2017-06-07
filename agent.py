import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

from collections import namedtuple

Hyperparameters = namedtuple('Hyperparameters', ['env', # OpenAI Gym environment id
                                                 'replay_buffer', # Replay buffer holding 50000 steps
                                                 'exploration_schedule', # Epsilon decay schedule
                                                 'num_nodes', # Number of nodes per hidden layer
                                                 'act_fns', # Activation functions for each layer
                                                 'optimizer', # Q-Function optimizer with given learning rate
                                                 'mean_window', # Window size to compute mean over last episodes
                                                 'min_t_solved', # Minimum episodes before considering a taask solved, should be >= mean_window
                                                 'min_mean_reward', # Minimum mean reward to consider a task solved
                                                 'minibatch_size', # Number of steps sampled from the replay buffer
                                                 'min_t_update', # Minimum episodes before minimizing
                                                 'update_period' # Number of episodes after which the target network gets updated
                                                 ])

cartpole_config = Hyperparameters(env='CartPole-v0',
                                  replay_buffer=ReplayBuffer(50000),
                                  # Create the schedule for exploration starting from 1 (every action is random) down to
                                  # 0.02 (98% of actions are selected according to values predicted by the model) in 10000 steps.
                                  exploration_schedule=LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02),
                                  num_nodes=[64],
                                  act_fns=[tf.nn.tanh],
                                  optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
                                  mean_window=100,
                                  min_t_solved=100,
                                  min_mean_reward=200,
                                  minibatch_size=32,
                                  min_t_update=1000,
                                  update_period=1000)

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
        for n_nodes, activation_fn in zip(config.num_nodes, config.act_fns):
            out = layers.fully_connected(out, num_outputs=n_nodes, activation_fn=activation_fn)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out



if __name__ == '__main__':
    # select configuration
    config = cartpole_config

    with U.make_session(8):
        # Create the environment
        env = gym.make(config.env)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=config.optimizer,
        )

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=config.exploration_schedule.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            config.replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > config.min_t_solved and np.mean(episode_rewards[-(config.mean_window+1):-1]) >= config.min_mean_reward
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > config.min_t_update:
                    obses_t, actions, rewards, obses_tp1, dones = config.replay_buffer.sample(config.minibatch_size)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % config.update_period == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-(config.mean_window+1):-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * config.exploration_schedule.value(t)))
                logger.dump_tabular()
