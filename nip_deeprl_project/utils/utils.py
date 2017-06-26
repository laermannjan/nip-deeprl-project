import os
import json

import gym
from gym.utils import atomic_write

def write_manifest(args, directory, name=None):
    fname = '{}manifest.json'.format('{}.'.format(name) if name is not None else '')
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with atomic_write.atomic_write(os.path.join(directory, fname)) as f:
        json.dump({
            'gym_version': gym.version.VERSION,
            'args': vars(args)
        }, f)

def get_last_run_number(directory):
    '''Returns name of subdir with highest integer as name.'''
    digit_subdirs = [int(sub) for sub in next(os.walk(directory))[1] if sub.isdigit()]
    return max(digit_subdirs if len(digit_subdirs) else [-1])
