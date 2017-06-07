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
                                                 ])

Configs =  {'cartpole_basic' : Hyperparameters(env='CartPole-v0',
                                               replay_buffer_size=50000,
                                               schedule_timesteps=10000,
                                               initial_p=1.0,
                                               final_p=0.02,
                                               exploration_schedule=LinearSchedule,
                                               exploration_fraction=0.1,
                                               num_nodes=[64],
                                               act_fns=[tf.nn.tanh],
                                               learning_rate=5e-4,
                                               optimizer=tf.train.AdamOptimizer,
                                               mean_window=100,
                                               min_t_solved=100,
                                               min_mean_reward=199,
                                               minibatch_size=32,
                                               learning_delay=1000,
                                               update_freq=1000,
                                               checkpoint_freq=10000,
                                               print_freq=10,
                                               max_timesteps=100000,
                                               gamma=1.0,
                                               train_freq=1,
                                               is_solved_func=1),
            }

