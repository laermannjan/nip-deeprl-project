import argparse
import os
import datetime

from baselines import logger
import baselines.common.tf_util as U

from configs import Configs
from project_framework.training import CustomTrainer



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
    parser.add_argument('--repeat',
                        action='store',
                        dest='repeat',
                        type=int,
                        default=1,
                        help='Amount of times an experiment should be repeated.')
    parser.add_argument('--enable-videos',
                        action='store_true',
                        dest='enable_videos',
                        help='Toggle to turn on video capturing.')
    parser.add_argument('--dir',
                        action='store',
                        type=str,
                        dest='root_dir',
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
    for i, config_name in enumerate(args.configs):
        logger.log('#'*35 + '\n' +
                   'Performing [{}/{}] experiments...\n'.format(i+1, len(args.configs)) +
                   '#'*35)
        for run in range(args.repeat):
            logger.log('::: Performing Run [{}/{}]...\n '.format(run+1, args.repeat))
            exp_name = '{}_{}_{}_{}'.format(args.env, config_name, args.exp_name, run)
            trainer = CustomTrainer(env_id=args.env,
                                    config_name=config_name,
                                    root_dir=args.root_dir,
                                    exp_name=exp_name,
                                    videos_enabled=args.enable_videos,
                                    num_cpu=args.num_cpu)
            trainer.train()
            U.reset()
