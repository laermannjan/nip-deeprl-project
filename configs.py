from enum import Enum
from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.layers as layers

from baselines.common.schedules import LinearSchedule

Hyperparameters = namedtuple('Hyperparameters', ['env', # OpenAI Gym environment id
                                                 'replay_buffer_size', # Replay buffer holding 50000 steps
                                                 'exploration_schedule', # Epsilon decay schedule
                                                 'exploration_fraction', # Alternative: fraction of entire learning period where p is annealed
                                                 'initial_p', # Initial percentage of random actions
                                                 'final_p', # Final percentage of random actions
                                                 'schedule_timesteps', # Number of steps until final_p is reached
                                                 'num_nodes', # Number of nodes per hidden layer
                                                 'act_fns', # Activation functions for each layer
                                                 'learning_rate', 
                                                 'optimizer', # Q-Function optimizer with given learning rate
                                                 'mean_window', # Window size to compute mean over last episodes
                                                 'min_t_solved', # Minimum episodes before considering a taask solved, should be >= mean_window
                                                 'min_mean_reward', # Minimum mean reward to consider a task solved
                                                 'minibatch_size', # Number of steps sampled from the replay buffer
                                                 'learning_delay', # Minimum episodes before minimizing
                                                 'update_freq', # Number of episodes after which the target network gets updated
                                                 'checkpoint_freq', # Number of episodes after which we save our system state
                                                 'print_freq', # How often we print a summary of our progress
                                                 'is_solved_func', # Define when a task is considered to be learned.
                                                 'max_timesteps', # Number of timesteps after which learning is aborted
                                                 'gamma', # Discount factor in bellman eq.
                                                 'train_freq', # Number of steps after which we update the model
                                                 'max_timesteps_ep', # Maximum timesteps per episode before *done*
                                                 'done_reward', # Reward when episode is done
                                                 ])

Configs =  {'cartpole_basic' : Hyperparameters(env='CartPole-v0',
                                               replay_buffer_size=50000,
                                               max_timesteps=100000,
                                               max_timesteps_ep=500,

                                               initial_p=1.0,
                                               final_p=0.02,

                                               exploration_fraction=0.1,

                                               exploration_schedule=LinearSchedule,
                                               schedule_timesteps=10000,

                                               num_nodes=[64],
                                               act_fns=[tf.nn.tanh],

                                               learning_rate=5e-4,
                                               optimizer=tf.train.AdamOptimizer,

                                               mean_window=100,
                                               min_t_solved=100,
                                               min_mean_reward=199,

                                               learning_delay=1000,
                                               minibatch_size=32,

                                               train_freq=1,
                                               update_freq=1000,
                                               gamma=1.0,

                                               done_reward=-100,

                                               checkpoint_freq=10000,
                                               print_freq=10,

                                               is_solved_func=1),

            'acrobot_basic' : Hyperparameters(env='Acrobot-v1',
                                              replay_buffer_size=50000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
                                              max_timesteps=2500000,
                                              max_timesteps_ep=5000,

                                              initial_p=1.0,
                                              final_p=0.1,

                                              # naive
                                              exploration_fraction=0.6,
                                              # custom
                                              exploration_schedule=LinearSchedule,
                                              schedule_timesteps=150000,

                                              num_nodes=[30, 30, 30],
                                              act_fns=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh],

                                              learning_rate=1e-3,
                                              optimizer=tf.train.AdamOptimizer,

                                              mean_window=100,
                                              min_t_solved=50000, # should be irrelevant
                                              min_mean_reward=-200, # accumulated reward negative for acrobot

                                              learning_delay=50000, # should be equal to epoche length, s.t. our replay buffer is full
                                              minibatch_size=100, # the smaller we choose, the more instable our system becomes
                                              # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
                                              # minibatch_size * train_freq = replay_buffer_size * 2
                                              train_freq=250, # = 50000/200 train mainDQN  200 times per update of targetDQN
                                              update_freq=50000,
                                              gamma=0.99,

                                              done_reward=30,

                                              checkpoint_freq=10000,
                                              print_freq=10,

                                              is_solved_func=1
            ),
}

