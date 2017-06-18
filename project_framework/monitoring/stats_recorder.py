import json
import os
import time
import numpy as np
import logging

from gym import error
from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np

from baselines.common.misc_util import RunningAvg
import baselines.common.tf_util as U
from baselines import logger

class StatsRecorder(object):
    def __init__(self, directory, file_prefix, autoreset=False, env_id=None, mean_window_size=100):
        self.autoreset = autoreset
        self.env_id = env_id

        self.initial_reset_timestamp = None
        self.directory = directory
        self.file_prefix = file_prefix
        self.episode_lengths = []
        self.mean_episode_lengths = []
        self.episode_rewards = []
        self.mean_episode_rewards = []
        self.episode_errors = []
        self.episode_types = []
        self.episode_lrates = []
        self._type = 't'
        self.timestamps = []
        self.steps = None
        self.total_steps = 0
        self.rewards = None
        self.mean_window_size = mean_window_size
        self.errors = []
        self.losses = []
        self.episode_explorations = []


        self.done = None
        self.closed = False

        filename = '{}.stats.json'.format(self.file_prefix)
        self.path = os.path.join(self.directory, filename)


    def update_errors(self, errs):
        self.errors.append(errs)
        self.losses.append(U.huber_loss(errs))

    def update_exploration(self, eps):
        self.episode_explorations.append(eps)
    
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in ['t', 'e']:
            raise error.Error('Invalid episode type {}: must be t for training or e for evaluation', type)
        self._type = type

    def before_step(self, action):
        assert not self.closed

        if self.done:
            raise error.ResetNeeded('''Trying to step env which is already done.
            While monitor is active for {}, you cannot call reset() unless episode is over, i.e. gym returns done=True.'''
                                    .format(self.env_id))
        if self.steps is None:
            raise error.ResetNeeded('''Trying to step before reset. Call 'env.reset()' before initial step.''')

    def after_step(self, obs, rew, done, info):
        self.steps += 1
        self.total_steps += 1
        self.rewards += rew
        self.done = done

        if done:
            self.save_complete()

            if self.autoreset:
                self.before_reset()
                self.after_reset(obs)

    def before_reset(self):
        assert not self.closed

        if self.done is not None and not self.done and self.steps > 0:
            raise error.Error('''Tried to reset Environment which is not done.
            While monitor is active for {}, you cannot call reset() unless episode is over, i.e. gym returns done=True.'''
                              .format(self.env_id))

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_reset(self, obs):
        self.steps = 0
        self.rewards = 0

    def record_td_error(self, td_error):
        # self.td_errors.append(td_error)
        # self.losses.append()
        pass

    def save_complete(self):
        if self.steps is not None:
            self.timestamps.append(time.time())
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(float(self.rewards))
            # Averaging
            self.mean_episode_lengths.append(np.mean(self.episode_lengths[-self.mean_window_size:]))
            self.mean_episode_rewards.append(np.mean(self.episode_rewards[-self.mean_window_size:]))

            self.episode_types.append(self._type)

    def close(self):
        self.flush()
        self.closed = True

    def flush(self):
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'mean_window_size': self.mean_window_size,
                'episode_lengths': self.episode_lengths,
                'mean_episode_lengths': self.mean_episode_lengths,
                'episode_rewards': self.episode_rewards,
                'mean_episode_rewards': self.mean_episode_rewards,
                'episode_types': self.episode_types,
            }, f, default=json_encode_np)
