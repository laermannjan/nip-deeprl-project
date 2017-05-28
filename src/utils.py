from enum import Enum
import logging
import tensorflow as tf
from gym import wrappers

class EpsilonStrategy(Enum):
    GREEDY = 1

def logger_setup(env_id, log_root_dir='../logging/', log_level=logging.INFO):
    import datetime

    timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')
    log_file = '{}{}_{}.log'.format(
        log_root_dir,
        env_id,
        timestamp)

    formatter = logging.Formatter('[%(asctime)s] (%(levelname)s): %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger

def train_agent(agent, env, monitor_root_dir=None):
    if monitor_root_dir:
        monitor_dir = '{}{}/training'.format(monitor_root_dir, timestamp)
        env = wrappers.Monitor(env, monitor_dir)
        logger.info('Monitor output saved here: {}'.format(monitor_dir))

    with tf.Session() as sess:
        agent.learn(env, sess)

