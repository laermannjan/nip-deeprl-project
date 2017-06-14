import argparse
import os
import datetime

from baselines import logger
import baselines.common.tf_util as U

from configs import Configs



if __name__ == '__main__':
    # get rid of TF warnings in the beginning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser(prog='custom_train',
                                     description='Train agent to play an OpenAI gym env.')
    parser.add_argument('env',
                        action='store',
                        choices=['Cartpole-v0', 'LunarLander-v2', 'Acrobot-v1'],
                        metavar='ENV',
                        type=str,
                        help='OpenAI Gym Id of the environment.')
    parser.add_argument('configs',
                        action='store',
                        nargs='+',
                        choices=Configs.keys(),
                        metavar='CONFIG',
                        type=str,
                        help='Name of config from configs.py.')
    parser.add_argument('--simple',
                        action='store_true',
                        help='Use simple trainer instead of custom.')
    parser.add_argument('--dir',
                        action='store',
                        type=str,
                        dest='pickle_root',
                        default='experiments',
                        help='Root directory for this experiment.')
    parser.add_argument('--cores',
                        action='store',
                        type=int,
                        dest='num_cpu',
                        default=8,
                        help='Number of cpu cores to be used for training.')
    parser.add_argument('--name',
                        action='store',
                        type=str,
                        dest='exp_name',
                        default=datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'),
                        help='Name of this experiment.')
    args = parser.parse_args()
    if args.simple:
        from train import train
    else:
        from custom_train import train as train

    for i, config_name in enumerate(args.configs):
        logger.log('#'*35 + '\n' +
                   'Performing [{}/{}] experiments...\n'.format(i+1, len(args.configs)) +
                   '#'*35)
        train(args.env, config_name, args.pickle_root, args.exp_name, args.num_cpu,run_num=i+1)
        U.reset()
