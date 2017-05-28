from enum import Enum
import logging
import tensorflow as tf
from gym import wrappers

class EpsilonStrategy(Enum):
    GREEDY = 1

def logger_setup(env_id, log_root_dir='../logging/', log_level=logging.INFO):
    import datetime
    import os
    
    if not os.path.exists(log_root_dir):
        os.makedirs(log_root_dir)
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

def train_agent(agent, env, session, monitor_root_dir=None, logger=None):
    if monitor_root_dir:
        monitor_dir = '{}{}/training'.format(monitor_root_dir, timestamp)
        env = wrappers.Monitor(env, monitor_dir)
        logger.info('Monitor output saved here: {}'.format(monitor_dir))

    agent.learn(env, session)

def compare_agents(agent1, agent2, env, session, logger=None):
    logger.info('Comparing Agent1 ({}) with Agent2 ({})'\
                .format((agent1.__class__, agent1.config),
                        (agent2.__class__, agent2.config)))
    r1, r2 = [], []
    n_iter = 200

    for i in range(n_iter):
        r1.append(agent1.play(env, session))
        r2.append(agent2.play(env, session))
    r1_avg = sum(r1)/n_iter
    r2_avg = sum(r2)/n_iter
    if logger is not None:
        logger.debug('Agent1 performances: {}'.format(r1))
        logger.debug('Agent2 performances: {}'.format(r2))
        logger.info('Agent1 avg. performance: {}'.format(r1_avg))
        logger.info('Agent2 avg. performance: {}'.format(r2_avg))
