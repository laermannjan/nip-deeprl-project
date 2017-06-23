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
