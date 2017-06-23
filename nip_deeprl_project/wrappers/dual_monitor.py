import time
import numpy as np
import os
import json

import gym
from gym import error, version
from gym.monitoring import video_recorder
from gym.utils import atomic_write

class DualMonitor(gym.Wrapper):
    def __init__(self, env, directory, run=None, video_callable=None, write_upon_reset=False):
        """Adds two qunatities to info returned by every step:
            num_steps: int
                Number of steps takes so far
            rewards: [float]
                All the cumulative rewards for the episodes completed so far.
        """
        super().__init__(env)
        # current episode state
        self._current_reward = None
        self._num_steps = None
        # temporary monitor state that we do not save
        self._time_offset = None
        self._total_steps = None
        self._episode_id = None
        # monitor state
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_end_times = []
        self._timestep_explorations = []
        self._epoche_td_errors = []
        # extras
        self._video_recorder = None
        self._augmented_reward = None

        self.write_upon_reset = write_upon_reset
        self._start(directory, video_callable, run)

    def _start(self, directory, video_callable, run):
        self.directory = os.path.abspath(os.path.join(directory, str(os.getpid())))
        # Create directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)
        # Turn video capturing on if needed and set schedule
        if video_callable is True:
            video_callable = default_video_schedule
        elif video_callable is False:
            video_callable = disable_videos
        elif not callable(video_callable):
            raise error.Error('You must provide a function, True, or False for video_callable, not {}: {}'
                              .format(type(video_callable), video_callable))
        self.video_callable = video_callable
        # Set common file infix
        self.file_infix = 'run_{:02}'.format(run if run is not None else 0)

    def _reset(self):
        obs = self.env.reset()
        # recompute temporary state if needed
        if self._time_offset is None:
            self._time_offset = time.time()
            if len(self._episode_end_times) > 0:
                self._time_offset -= self._episode_end_times[-1]
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)
        # update monitor state
        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._episode_end_times.append(time.time() - self._time_offset)
        if self._episode_id is None:
            self._episode_id = 0
        else:
            self._episode_id += 1
        # reset episode state
        self._current_reward = 0
        self._num_steps = 0
        # end video recorder and start a new one
        self._reset_video_recorder()
        # write to disk if needed
        self._flush()

        return obs

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self._augmented_reward:
            rew = self._augmented_reward
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        info['steps'] = self._total_steps
        info['rewards'] = self._episode_rewards

        self._video_recorder.capture_frame()
        return (obs, rew, done, info)

    def get_state(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'episode_data': {
                'episode_rewards': self._episode_rewards,
                'episode_lengths': self._episode_lengths,
                'episode_end_times': self._episode_end_times,
                'epoche_td_errors': self._epoche_td_errors,
                'timestep_explorations': self._timestep_explorations,
                'initial_reset_time': 0,
            }
        }

    def set_state(self, state):
        assert state['env_id'] == self.env.unwrapped.spec.id
        ed = state['episode_data']
        self._episode_rewards = ed['episode_rewards']
        self._episode_lengths = ed['episode_lengths']
        self._episode_end_times = ed['episode_end_times']
        self._timestep_explorations = ed['timestep_explorations']
        self._epoche_td_errors = ed['epoch_td_errors']


    def _reset_video_recorder(self):
        # Close all recorders
        if self._video_recorder:
            self._video_recorder.close()

        self._video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory, 'video.{}.episode_{:06}'
                                   .format(self.file_infix, self._episode_id)),
            enabled=self._video_enabled()
            )
        self._video_recorder.capture_frame()


    def _flush(self, force=False):
        if self.write_upon_reset or force:
            # Write stats file
            fname = 'stats.{}.npz'.format(self.file_infix)
            np.savez(os.path.join(self.directory, fname),
                     episode_lengths=np.array(self._episode_lengths),
                     episode_rewards=np.array(self._episode_rewards),
                     episode_end_times=np.array(self._episode_end_times),
                     td_errors=np.array(self._epoche_td_errors)
                     )
            # Write manifest
            fname = 'manifest.json'
            with atomic_write.atomic_write(os.path.join(self.directory, fname)) as f:
                json.dump({
                    'env_id': self.env.unwrapped.spec.id,
                    'gym_version': version.VERSION
                }, f)


    def record_exploration(self, exploration):
        self._timestep_explorations.append(exploration)

    def record_td_errors(self, td_errors):
        self._epoche_td_errors.append(td_errors)

    def set_augmented_reward(self, reward):
        self._augmented_reward = reward

    def close(self):
        self._video_recorder.close()
        self._flush(force=True)

    def _video_enabled(self):
        return self.video_callable(self._episode_id)

def default_video_schedule(episode_id):
    '''Returns true if episode_id is a perfect cube or a mulitple of 1000.'''
    if episode_id < 1000:
        return round(episode_id ** (1. / 3)) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

def disable_videos(episode_id):
    return False
