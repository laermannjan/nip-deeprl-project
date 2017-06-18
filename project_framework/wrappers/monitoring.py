import os, json, logging

import gym
from gym import error, version
from gym.monitoring import video_recorder
from gym.utils import atomic_write, closer
from gym.utils.json_utils import json_encode_np

from baselines import logger

from project_framework.monitoring import stats_recorder

FILE_PREFIX = 'LOG'
MANIFEST_PREFIX = FILE_PREFIX + '.manifest'

class Monitor(gym.Wrapper):
    def __init__(self, env, directory, video_callable=None, force=False, resume=False, write_upon_reset=False, uid=None, mode=None):
        super(Monitor, self).__init__(env)

        self.videos = []

        self.stats_recorder = None
        self.video_recorder = None
        self.enabled = False
        self.episode_id = 0
        self._monitor_id = None
        self.env_semantics_autoreset = env.metadata.get('semantics.autoreset')

        self._act_func = None
        self._train_func = None
        self._update_func = None

        self.augmented_reward = None
        self.config_name = None

        self._start(directory, video_callable, force, resume,
                    write_upon_reset, uid, mode)

    def _step(self, action):
        self._before_step(action)
        obs, rew, done, info = self.env.step(action)
        rew = self.reward(rew, done)
        done = self._after_step(obs, rew, done, info)

        return obs, rew, done, info

    def _reset(self):
        self._before_reset()
        obs = self.env.reset()
        self._after_reset(obs)

        return obs

    def _close(self):
        super(Monitor, self)._close()

        # _monitor will not be set if super(Monitor, self).__init__ raises, this check prevents a confusing error message
        if getattr(self, '_monitor', None):
            self.close()

    def _start(self, directory, video_callable=None, force=False, resume=False,
               write_upon_reset=False, uid=None, mode=None):
        """Start monitoring.

        Args:
            directory (str): A per-training run directory where to record stats.
            video_callable (Optional[function, False]): function that takes in the index of the episode and outputs a boolean, indicating whether we should record a video on this episode. The default (for video_callable is None) is to take perfect cubes, capped at 1000. False disables video recording.
            force (bool): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
            resume (bool): Retain the training data already in this directory, which will be merged with our new data
            write_upon_reset (bool): Write the manifest file on each reset. (This is currently a JSON file, so writing it is somewhat expensive.)
            uid (Optional[str]): A unique id used as part of the suffix for the file. By default, uses os.getpid().
            mode (['evaluation', 'training']): Whether this is an evaluation or training episode.
        """
        env_id = self.env.spec.id
        if not os.path.exists(directory):
            # logger
            os.makedirs(directory, exist_ok=True)

        if video_callable is None:
            video_callable = capped_cubic_video_schedule
        elif video_callable == False:
            video_callable = disable_videos
        elif not callable(video_callable):
            raise error.Error('You must provide a function, None, or False for video_callable, not {}: {}'
                              .format(type(video_callable), video_callable))
        self.video_callable = video_callable

        # Check if there are previous recordings
        if force:
            clear_monitor_files(directory)
        elif not resume:
            training_manifests = detect_training_manifests(directory)
            if len(training_manifests) > 0:
                raise error.Error('''Trying to write to monitor directory {} with existing monitor files: {}.

 You should use a unique directory for each training run, or use 'force=True' to automatically clear previous monitor files.'''
                                  .format(directory, ', '.join(training_manifests[:5])))

        self._monitor_id = monitor_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)
        self.file_prefix = FILE_PREFIX
        self.file_infix = '{}_{}'.format(self._monitor_id, uid if uid else os.getpid())

        self.stats_recorder = stats_recorder.StatsRecorder(directory, '{}.episode_batch.{}'
                                                           .format(self.file_prefix, self.file_infix))

        self.write_upon_reset = write_upon_reset

        if mode is not None:
            self._set_mode(mode)

    def _flush(self, force=False):
        '''Flush all monitor stats to disk.'''
        if not self.write_upon_reset and not force:
            return

        self.stats_recorder.flush()

        path = os.path.join(self.directory, '{}.manifest.{}.json'
                            .format(self.file_prefix, self.file_infix))
        with atomic_write.atomic_write(path) as f:
            json.dump({
                'stats': os.path.basename(self.stats_recorder.path),
                'videos': [(os.path.basename(v), os.path.basename(m))
                           for v, m in self.videos],
                'env_info': self._env_info(),
                'config': self.config_name,

            }, f, default=json_encode_np)

    def close(self):
        '''Flush all monitor stats to disk and close all open rendering windows.'''
        if not self.enabled:
            return
        self.stats_recorder.close()
        if self.video_recorder is not None:
            self._close_video_recorder()
        self._flush(force=True)

        monitor_closer.unregister(self._monitor_id)
        self.enabled = False

        # log here where files where put

    def _record_errors(self, errs):
        self.stats_recorder.update_errors(errs)

    def _record_exploration(self, eps):
        self.stats_recorder.update_exploration(eps)

    def reward(self, reward, done):
        if self.augmented_reward is None:
            return reward
        return self._reward(reward, done)

    def _reward(self, reward, done):
        if done:
            return self.augmented_reward
        return reward

    def set_augmented_reward(self, reward):
        self.augmented_reward = reward

    def register_graph(self, act, train, update):
        # should have some checking here
        self.register_act(act)
        self.register_train(train)
        self.register_update(update)

    def register_act(self, act_func):
        self._act_func = act_func

    def register_train(self, train_func):
        self._train_func = train_func

    def register_update(self, update_func):
        self._update_func = update_func

    def act(self, obs, update_eps):
        assert self._act_func
        self._record_exploration(update_eps)
        self._act_func(obs, update_eps=update_eps)

    def train(self, obs_t, action, reward, obs_tp1, done, weight):
        td_errors = self._train_func(obs_t, action, reward, obs_tp1, done, weight)
        self._record_errors(td_errors)
        return td_errors

    def update(self):
        self._update_func()

    def _set_mode(self, mode):
        if mode == 'evaluation':
            type = 'e'
        elif mode == 'training':
            type = 't'
        else:
            raise error.Error('Invalid mode {}: must be "training" or "evaluation"', mode)
        self.stats_recorder.type = type

    def _before_step(self, action):
        if not self.enabled:
            return
        self.stats_recorder.before_step(action)

    def _after_step(self, obs, rew, done, info):
        if not self.enabled:
            return done

        if done and self.env_semantics_autoreset:
            self._reset_video_recorder()
            self.episode_id += 1
            self._flush()

        if info.get('true_reward', None):  # Some semi-supervised envs modify rewards
            rew = info['true_reward']

        self.stats_recorder.after_step(obs, rew, done, info)
        self.video_recorder.capture_frame()

        return done

    def _before_reset(self):
        if not self.enabled:
            return
        self.stats_recorder.before_reset()

    def _after_reset(self, obs):
        if not self.enabled:
            return

        # Reset stats
        self.stats_recorder.after_reset(obs)

        self._reset_video_recorder()

        self.episode_id += 1

        self._flush()

    def _reset_video_recorder(self):
        # Close all recorders
        if self.video_recorder:
            self._close_video_recorder()

        # TODO: calculate better episode_id on merge (resume)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory, '{}.video.{}.video{:06}'
                                   .format(self.file_prefix, self.file_infix, self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled()
        )
        self.video_recorder.capture_frame()

    def _close_video_recorder(self):
        self.video_recorder.close()
        if self.video_recorder.functional:
            self.videos.append((self.video_recorder.path, self.video_recorder.metadata_path))

    def _video_enabled(self):
        return self.video_callable(self.episode_id)

    def _env_info(self):
        env_info = {
            'gym_version': version.VERSION
        }
        if self.env.spec:
            env_info['env_id'] = self.env.spec.id
        return env_info

    def __del__(self):
        # Making sure all recorders are closed when the garbage collector comes
        self.close()

    def get_total_steps(self):
        return self.stats_recorder.total_steps

    def get_episode_rewards(self):
        return self.stats_recorder.episode_rewards

    def get_mean_episode_rewards(self):
        return self.stats_recorder.mean_episode_rewards

    def get_current_mean_reward(self):
        return self.stats_recorder.mean_episode_rewards[-1]

    def get_episode_lengths(self):
        return self.stats_recorder.episode_lengths

    def get_mean_episode_lengths(self):
        return self.stats_recorder.mean_episode_lengths

class ActionMonitor(gym.ActionWrapper):
    def _action(self, action):
        pass


def detect_training_manifests(training_dir, files=None):
    if files is None:
        files = os.listdir(training_dir)
    return [os.path.join(training_dir, f) for f in files if f.startswith(MANIFEST_PREFIX + '.')]

def detect_monitor_files(training_dir):
    files = os.listdir(training_dir)
    return [os.path.join(training_dir, f) for f in files if f.startswith(FILE_PREFIX + '.')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return

    for file in files:
        os.unlink(file)


def capped_cubic_video_schedule(episode_id):
    '''Returns true if episode_id is a perfect cube or a mulitple of 1000.'''
    if episode_id < 1000:
        return int(round(episode_id ** (1. / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

def disable_videos(episode_id):
    return False

monitor_closer = closer.Closer()
