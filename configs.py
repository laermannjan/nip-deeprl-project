Configs = {
    'AB_basic': {
        'env': 'Acrobot-v1',
        'arch': [64, 64, 64],
        'gamma': 0.99,
        'lr':5e-3,
        'num_steps': 500*6000,
        'num_episodes': 500*15,
        'max_episode_steps': 3000, 
        'schedule_timesteps': 3000*100,
        'initial_p': 1.0,
        'final_p': 0.1,
        'learning_freq': 1,
        'target_update_freq': 3000*30,
        'replay_buffer_size': 3000*20,
        'batch_size': 100,
        'augmented_reward': None,
        'save_freq': 3000*20,
    },
    'LL_basic': {
        'env': 'LunarLander-v2',
        'arch': [50, 50, 50],
        'gamma': 0.99,
        'lr':5e-3,
        'num_steps': 500*4000,
        'num_episodes': 500*4,
        'max_episode_steps': 500,
        'schedule_timesteps': 500*500,
        'initial_p': 1.0,
        'final_p': 0.02,
        'learning_freq': 1,
        'target_update_freq': 500*30,
        'replay_buffer_size': 500*20,
        'batch_size': 100,
        'augmented_reward': None,
        'save_freq': 500*20,
    },
    'LL_e1_short_in': {
        'env': 'LunarLander-v2',
        'final_p': 0.01,
        'schedule_timesteps': 500*300,
        'arch': [128, 256, 512]
    },
    'LL_e1_short_eq': {
        'env': 'LunarLander-v2',
        'final_p': 0.01,
        'schedule_timesteps': 500*300,
        'arch': [256, 256, 256]
    },
    'LL_e1_short_de': {
        'env': 'LunarLander-v2',
        'final_p': 0.01,
        'schedule_timesteps': 500*300,
        'arch': [256, 192, 128]
    },
    'LL_e1_long_in': {
        'env': 'LunarLander-v2',
        'final_p': 0.01,
        'schedule_timesteps': 500*800,
        'arch': [128, 192, 256]
    },
    'LL_e1_long_eq': {
        'env': 'LunarLander-v2',
        'final_p': 0.01,
        'schedule_timesteps': 500*800,
        'arch': [128, 128, 128]
    },
    'LL_e1_long_de': {
        'env': 'LunarLander-v2',
        'final_p': 0.01,
        'schedule_timesteps': 500*800,
        'arch': [256, 192, 128]
    },
    'LL_e10_short_in': {
        'env': 'LunarLander-v2',
        'final_p': 0.1,
        'schedule_timesteps': 500*300,
        'arch': [128, 192, 256]
    },
    'LL_e10_short_eq': {
        'env': 'LunarLander-v2',
        'final_p': 0.1,
        'schedule_timesteps': 500*300,
        'arch': [128, 128, 128]
    },
    'LL_e10_short_de': {
        'env': 'LunarLander-v2',
        'final_p': 0.1,
        'schedule_timesteps': 500*300,
        'arch': [256, 192, 128]
    },
    'LL_e10_long_in': {
        'env': 'LunarLander-v2',
        'final_p': 0.1,
        'schedule_timesteps': 500*800,
        'arch': [128, 192, 256]
    },
    'LL_e10_long_eq': {
        'env': 'LunarLander-v2',
        'final_p': 0.1,
        'schedule_timesteps': 500*800,
        'arch': [128, 128, 128]
    },
    'LL_e10_long_de': {
        'env': 'LunarLander-v2',
        'final_p': 0.1,
        'schedule_timesteps': 500*800,
        'arch': [256, 192, 128]
    },
    'AB_e1_short_in': {
        'env': 'Acrobot-v1',
        'final_p': 0.01,
        'schedule_timesteps': 3000*100,
        'arch': [128, 192, 256]
    },
    'AB_e1_short_eq': {
        'env': 'Acrobot-v1',
        'final_p': 0.01,
        'schedule_timesteps': 3000*100,
        'arch': [128, 128, 128]
    },
    'AB_e1_short_de': {
        'env': 'Acrobot-v1',
        'final_p': 0.01,
        'schedule_timesteps': 3000*100,
        'arch': [256, 192, 128]
    },
    'AB_e1_long_in': {
        'env': 'Acrobot-v1',
        'final_p': 0.01,
        'schedule_timesteps': 5000*300,
        'arch': [128, 192, 256]
    },
    'AB_e1_long_eq': {
        'env': 'Acrobot-v1',
        'final_p': 0.01,
        'schedule_timesteps': 3000*300,
        'arch': [128, 128, 128]
    },
    'AB_e1_long_de': {
        'env': 'Acrobot-v1',
        'final_p': 0.01,
        'schedule_timesteps': 3000*300,
        'arch': [256, 192, 128]
    },
    'AB_e10_short_in': {
        'env': 'Acrobot-v1',
        'final_p': 0.1,
        'schedule_timesteps': 3000*100,
        'arch': [128, 192, 256]
    },
    'AB_e10_short_eq': {
        'env': 'Acrobot-v1',
        'final_p': 0.1,
        'schedule_timesteps': 5000*100,
        'arch': [128, 128, 128]
    },
    'AB_e10_short_de': {
        'env': 'Acrobot-v1',
        'final_p': 0.1,
        'schedule_timesteps': 3000*100,
        'arch': [256, 192, 128]
    },
    'AB_e10_long_in': {
        'env': 'Acrobot-v1',
        'final_p': 0.1,
        'schedule_timesteps': 3000*300,
        'arch': [128, 192, 256]
    },
    'AB_e10_long_eq': {
        'env': 'Acrobot-v1',
        'final_p': 0.1,
        'schedule_timesteps': 3000*300,
        'arch': [128, 128, 128]
    },
    'AB_e10_long_de': {
        'env': 'Acrobot-v1',
        'final_p': 0.1,
        'schedule_timesteps': 3000*300,
        'arch': [256, 192, 128]
    },
    'LL_e1_short_eq_prio': {
        'env': 'LunarLander-v2',
        'final_p': 0.01,
        'schedule_timesteps': 500*300,
        'arch': [128, 128, 128],
        'prioritized': True
    },
    'LL_e1_long_eq_prio': {
        'env': 'LunarLander-v2',
        'final_p': 0.01,
        'schedule_timesteps': 500*800,
        'arch': [128, 128, 128],
        'prioritized': True
    },
    'LL_e10_short_eq_prio': {
        'env': 'LunarLander-v2',
        'final_p': 0.1,
        'schedule_timesteps': 500*300,
        'arch': [128, 128, 128],
        'prioritized': True
    },
    'LL_e10_long_eq_prio': {
        'env': 'LunarLander-v2',
        'final_p': 0.1,
        'schedule_timesteps': 500*800,
        'arch': [128, 128, 128],
        'prioritized': True
    },
    'AB_e1_short_eq_prio': {
        'env': 'Acrobot-v1',
        'final_p': 0.01,
        'schedule_timesteps': 500*300,
        'arch': [128, 128, 128],
        'prioritized': True
    },
    'AB_e1_long_eq_prio': {
        'env': 'Acrobot-v1',
        'final_p': 0.01,
        'schedule_timesteps': 500*800,
        'arch': [128, 128, 128],
        'prioritized': True
    },
    'AB_e10_short_eq_prio': {
        'env': 'Acrobot-v1',
        'final_p': 0.1,
        'schedule_timesteps': 500*300,
        'arch': [128, 128, 128],
        'prioritized': True
    },
    'AB_e10_long_eq_prio': {
        'env': 'Acrobot-v1',
        'final_p': 0.1,
        'schedule_timesteps': 500*800,
        'arch': [128, 128, 128],
        'prioritized': True
    },
}
