from enum import Enum
from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.layers as layers

from baselines.common.schedules import LinearSchedule

Hyperparameters = namedtuple('Hyperparameters', [
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

Configs = {
    'LL_e10_short_eq' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.1,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*300,

        num_nodes=[256, 256, 256],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e10_short_de' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.1,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*300,

        num_nodes=[512, 256, 128],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e10_short_in' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.1,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*300,

        num_nodes=[128, 256, 512],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e10_long_eq' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.1,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*800,

        num_nodes=[256, 256, 256],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e10_long_de' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.1,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*800,

        num_nodes=[512, 256, 128],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e10_long_in' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.1,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*800,

        num_nodes=[128, 256, 512],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    #### ####################################################################################################
    ## exploration decay down to 1%
    ##
    #### ####################################################################################################
    'LL_e1_short_eq' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.01,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*300,

        num_nodes=[256, 256, 256],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e1_short_de' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.01,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*300,

        num_nodes=[512, 256, 128],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e1_short_in' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.01,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*300,

        num_nodes=[128, 256, 512],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e1_long_eq' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.01,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*800,

        num_nodes=[256, 256, 256],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e1_long_de' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.01,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*800,

        num_nodes=[512, 256, 128],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    'LL_e1_long_in' : Hyperparameters(
        replay_buffer_size=10000, #TODO: maybe calculate dynamically: steps_episode * episodes_training * num_epochs
        max_timesteps=500*6000,
        max_timesteps_ep=500,

        initial_p=1.0,
        final_p=0.01,

        # naive
        exploration_fraction=0.6,
        # custom
        exploration_schedule=LinearSchedule,
        schedule_timesteps=500*800,

        num_nodes=[128, 256, 512],
        act_fns=[tf.nn.relu, tf.nn.relu, tf.nn.relu],

        learning_rate=5e-3,
        optimizer=tf.train.AdamOptimizer,

        mean_window=100,
        min_t_solved=5000, # should be irrelevant
        min_mean_reward=250, # accumulated reward negative for acrobot

        learning_delay=5000, # should be equal to epoche length, s.t. our replay buffer is full
        minibatch_size=100, # the smaller we choose, the more instable our system becomes
        # instead of prio sampling, oversample replay buffer - needs to take minibatch size into account
        # minibatch_size * train_freq = replay_buffer_size * 2
        train_freq=50, # = 50000/200 train mainDQN  200 times per update of targetDQN
        update_freq=15000,
        gamma=0.99,

        done_reward=None,

        checkpoint_freq=10000,
        print_freq=50,

        is_solved_func=1
    ),
    
    
}
