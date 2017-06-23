import os
import time
import argparse
from gym import envs
from baselines.common.misc_util import boolean_flag
from nip_deeprl_project.training import train
from configs import Configs

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for OpenAI Gym games")
    # Environment
    parser.add_argument("--env", type=str, choices=[spec.id for spec in envs.registry.all()], default="LunarLander-v2", help="name of the game")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=40000, help="number of iterations between every target network update")
    parser.add_argument("--arch", type=int, default=[256]*3, nargs='+', help="number of nodes per layer for this model.")
    parser.add_argument("--gamma", type=float, default=0.99, help="DQN discount factor")
    parser.add_argument("--schedule-timesteps", type=int, default=150000, help="steps in which exploration fraction anneals from initial_p to final_p.")
    parser.add_argument("--initial-p", type=float, default=1.0, help="intial p")
    parser.add_argument("--final-p", type=float, default=0.01, help="final p")
    # Bells and whistles
    boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    boolean_flag(parser, "prioritized", default=False, help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized-alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default='data', help="directory in which training state and model should be saved.")
    parser.add_argument("--save-freq", type=int, default=1e6, help="save model once every time this many iterations are completed")
    boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
    # Augmentations
    parser.add_argument("--augmented-reward", type=int, default=None, help="augmented reward when environment is done.")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="augmented max length of each episode.")
    # Monitoring
    boolean_flag(parser, "write-upon-reset", default=False, help="if true results get written to disk after every episode. this might severly slow down the process.")
    boolean_flag(parser, "capture-videos", default=False, help="if true enables video capturing.")
    parser.add_argument("--run-num", default=None, help="identifies the run number if experiment is run mulitple times.")
    # Config
    parser.add_argument("--config", type=str, nargs='+', choices=Configs.keys(), default=None, help="define a config by name from configs.py which may overwrite other arguments")
    parser.add_argument("--repeat", type=int, default=1, help="number of times the same experiment is being repeated. if multiple configs are defined, each is being repeated individually.")
    return parser.parse_args()

def _load_config(args, config):
    for attr, val in Configs[config].items():
        setattr(args, attr, val)

def load_config(args, config):
    # Load defaults
    if Configs[config]['env'] == 'Cartpole-v0':
        _load_config(args, 'CP_basic')
    elif Configs[config]['env'] == 'LunarLander-v2':
        _load_config(args, 'LL_basi')
    elif Configs[config]['env'] == 'Acrobot-v1':
        _load_config(args, 'AB_basic')
    # Load modifications
    _load_config(args, config)
    # create subdir for config
    setattr(args, 'save_dir', os.path.join(args.save_dir, config))

if __name__ == '__main__':
    orig_args = parse_args()

    if orig_args.config is not None:
        for config in orig_args.config:
            args = orig_args
            load_config(orig_args, config)
            for _ in range(orig_args.repeat):
                # Train experiment
                train(args)
    else:
        args = orig_args
        setattr(args, 'save_dir', os.path.join(args.save_dir, 'TEST', str(time.time())))
        for _ in range(orig_args.repeat):
            train(args)
