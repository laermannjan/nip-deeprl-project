import argparse
import numpy as np
import os
import tempfile
import time
import itertools
import shutil

import tensorflow as tf

import gym
import scipy.misc
import math

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import baselines.common.tf_util as U
import tensorflow.contrib.layers as layers
from baselines.common.schedules import LinearSchedule
from baselines.common.misc_util import (
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
)

from nip_deeprl_project.wrappers import DualMonitor
from nip_deeprl_project.utils import write_manifest, build_graph_softmax

from skimage.measure import block_reduce
from skimage.color import rgb2gray 


MODELS_DIR = 'models'
PICKLE_DIR = 'agents'



def make_env(game_name, args, **kwargs):
    env = gym.make(game_name)
    augmented_env = env
    for attr, val in kwargs.items():
        if val is not None:
            setattr(augmented_env, attr, val)
    monitored_env = DualMonitor(augmented_env,
                                directory=args.save_dir,
                                write_upon_reset=args.write_upon_reset,
                                video_callable=args.capture_videos,
                                write_freq=args.write_freq,
                                image=args.image)
    return monitored_env


def maybe_save_model(savedir, state, pickle_name=None):
    """This function checkpoints the model and state of the training algorithm."""
    if savedir is None:
        return
    start_time = time.time()
    model_dir = os.path.join(savedir, MODELS_DIR, "model.step{}".format(state["num_iters"]))
    U.save_state(os.path.join(model_dir, "saved"))
    shutil.make_archive(model_dir,
                        'xztar',
                        model_dir)

    shutil.rmtree(model_dir)
    if pickle_name is not None:
        fname = '{}.pkl.zip'.format(pickle_name)
        if not os.path.exists(os.path.join(savedir, PICKLE_DIR)):
            os.makedirs(os.path.join(savedir, PICKLE_DIR), exist_ok=True)
        relatively_safe_pickle_dump(state, os.path.join(savedir, PICKLE_DIR, fname), compression=True)
        logger.log('Saved agent as {}.'.format(fname))
    logger.log("Saved model in {} seconds\n".format(time.time() - start_time))


def maybe_load_model(savedir):
    # TODO: untested and probably broken
    """Load model if present at the specified path."""
    if savedir is None:
        return

    state_path = os.path.join(os.path.join(savedir, 'training_state.pkl.zip'))
    found_model = os.path.exists(state_path)
    if found_model:
        state = pickle_load(state_path, compression=True)
        model_dir = "model.step{}".format(state["num_iters"])
        os.mkdir(os.path.join(savedir, model_dir))
        shutil.unpack_archive(os.path.join(savedir, MODEL_DIR, '{}.tar.xz'.format(model_dir)),
                              'xztar',
                              os.path.join(savedir, MODEL_DIR, model_dir))
        U.load_state(os.path.join(savedir, model_dir, "saved"))
        shutil.rmtree(os.path.join(save_dir, model_dir))
        logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
        return state

#def rgb2gray(img):
#    return np.dot(img, [0.299, 0.587, 0.114])[..., None]

def train(args):
    savedir = args.save_dir
    # Create and seed the env.
    env = make_env(args.env, args, _max_episode_steps=args.max_episode_steps)


    if args.seed > 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)

    with U.make_session(8) as sess:
        # Create training graph and replay buffer
        if args.image:
            ds_blocksize = (6,6,1)  # downsampling blocksize
            ds_dims = [math.ceil(i/d)
                       for i, d in zip(env.render('rgb_array').shape,
                                       ds_blocksize)]
            ds_dims[-1] = args.receptive_field_depth
            model = deepq.models.cnn_to_mlp(args.conv_arch, args.arch)
            mop = lambda name: U.BatchInput(ds_dims, name=name) # Unit8Input is optimized int8 input for GPUs
        else:
            model = deepq.models.mlp(args.arch)
            mop = lambda name: U.BatchInput(env.observation_space.shape, name=name)

        act, train, update_target, debug = deepq.build_train(
            make_obs_ph= mop,
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-8), # often 1e-4 for atari games, why?
            # -> epsilon is a const to prevent deviding by 0. formula is something like lr * x / (sqrt(x) +epsilon)
            # this may also depend on precision and maybe atari games were trained with half precision?!
            gamma=args.gamma,
            grad_norm_clipping=None if args.grad_clip is None else args.grad_clip, # was 10, why? -> clipping helps to keeps gradients under control. not sure why this is favored over L2 norm here.
            double_q=args.double_q
        )

        approximate_num_iters = args.num_steps if args.num_episodes is None else args.num_episodes * env.env._max_episode_steps / 2.
        approximate_num_iters /= 4
        exploration = LinearSchedule(schedule_timesteps=args.schedule_timesteps,
                                     initial_p=args.initial_p,
                                     final_p=args.final_p)
        if args.prioritized:
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(approximate_num_iters, initial_p=args.prioritized_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(args.replay_buffer_size)

        U.initialize()
        update_target()
        num_iters = 0

        state = maybe_load_model(savedir)
        # if state is not None:
        #     num_iters, replay_buffer = state["num_iters"], state["replay_buffer"],
        #     # TODO: implement set_state in monitoring.py
        #     env.set_state(state["monitor_state"])

        start_time, start_steps = None, None
        steps_per_episode = RunningAvg(0.999)
        episode_time_est = RunningAvg(0.999)
        if args.image:
            steps_per_episode = RunningAvg(0.9)
            episode_time_est = RunningAvg(0.9)

        best_mean_rew = -float('inf')

        if args.image:
            env.reset()
            obs = block_reduce( # downsampling
                np.dstack( # stacking images as quasi-channels
                    [rgb2gray(env.render('rgb_array')) # grayscaling
                     for _ in [env.step(env.action_space.sample())] * args.receptive_field_depth]),
                ds_blocksize,
                func=np.mean # shouldn't this be np.max?
            )
        else:
            obs = env.reset()

        # Main trianing loop
        for num_iters in itertools.count(num_iters):
            pickle_this, pickle_name = False, None
            # Take action and store transition in the replay buffer.
            update_eps = exploration.value(num_iters)
            if args.softmax:
                action = act(np.array(obs)[None], softmax=args.softmax, update_eps=update_eps)[0]
            else:
                action = act(np.array(obs)[None], update_eps=update_eps)[0]

            env.record_exploration(update_eps)
            if args.image:
                # repeat action for k steps but don't record them.
                new_obs = []
                rew = 0
                for _, new_rew, done, info in [env.step(action)] * args.frameskip:
                    new_pic = rgb2gray(env.render('rgb_array'))
                    new_obs.append(new_pic)
                    rew += new_rew
                    if done:
                        for i in range(args.frameskip - len(new_obs)):
                            new_obs.append(new_pic)
                        break

                new_obs = block_reduce( # downsampling
                    np.dstack(new_obs),# stacking images as quasi-channels
                    ds_blocksize,
                    func=np.mean # shouldn't this be np.max?
                )
            else:
                new_obs, rew, done, info = env.step(action)


            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs


            if done:
                if np.mean(info["rewards"][-100:]) > best_mean_rew:
                    best_mean_rew = np.mean(info["rewards"][-100:])
                if args.image:
                    env.reset()
                else:
                    obs = env.reset()
                if info["episodes"] + 1 == args.num_episodes:
                    env._reset_video_recorder(force=True)

            # if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and # num_iters > args.min_t_learning
            if (num_iters > args.learning_delay and
                    num_iters % args.learning_freq == 0):
                for i_iter in range(args.num_samples):
                    # Sample a bunch of transitions from replay buffer
                    if args.prioritized:
                        experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                        weights = np.ones_like(rewards)
                    # Minimize the error in Bellman's equation and compute TD-error
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                    env.record_td_errors(td_errors)
                    # Update the priorities in the replay buffer
                    if args.prioritized:
                        new_priorities = np.abs(td_errors) + args.prioritized_eps
                        replay_buffer.update_priorities(batch_idxes, new_priorities)
            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if start_time is not None and done:
                steps_per_episode.update(info['steps'] - start_steps)
                episode_time_est.update(time.time() - start_time)
            if start_time is None or done:
                start_time, start_steps = time.time(), info["steps"]

            # Save the model and training state.
            pickle_dict = {
                    "replay_buffer": replay_buffer,
                    "num_iters": num_iters,
                    "monitor_state": env.get_state()
            }


            if num_iters > 0 and num_iters % args.save_freq == 0:
                maybe_save_model(args.save_dir,
                                 pickle_dict,
                                 pickle_name=None)
                if best_mean_rew < np.mean(info["rewards"][-100:]):
                    maybe_save_model(args.save_dir,
                                     pickle_dict,
                                     pickle_name='best_mean_rew')
            if info["steps"] == args.num_steps:
                maybe_save_model(args.save_dir,
                                 pickle_dict,
                                 pickle_name='final_step')
            if args.num_episodes is not None and info["episodes"] == args.num_episodes and done:
                maybe_save_model(args.save_dir,
                                 pickle_dict,
                                 pickle_name='final_episode')

            if args.num_episodes is None:
                if info["steps"] > args.num_steps:
                    break
            elif info['episodes'] > args.num_episodes:
                break

            if done:
                if args.num_episodes is not None:
                    eps_left = args.num_episodes - info["episodes"]
                    completion = np.round(info["episodes"] / args.num_episodes, 1)
                else: 
                    steps_left = args.num_steps - info["steps"]
                    completion = np.round(info["steps"] / args.num_steps, 1)

                logger.record_tabular("% completion", completion)
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("episodes", info["episodes"])
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-100:]))
                logger.record_tabular("length avg", steps_per_episode._value if steps_per_episode._value is not None else "calculating...")
                logger.record_tabular("exploration", exploration.value(num_iters))
                if args.prioritized:
                    logger.record_tabular("max priority", replay_buffer._max_priority)
                logger.dump_tabular()
                logger.log()
                fps_estimate = (float(steps_per_episode._value) / (float(episode_time_est) + 1e-6) \
                                if steps_per_episode._value is not None else 1)
                if args.num_episodes is not None:
                    eta = int(eps_left * float(steps_per_episode._value) / fps_estimate \
                          if steps_per_episode._value is not None else env.env._max_episode_steps)
                else:
                    eta = int(steps_left / fps_estimate)
                logger.log("ETA: " + pretty_eta(eta))
                logger.log()
