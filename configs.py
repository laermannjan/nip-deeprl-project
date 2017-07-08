Configs = {
    'AB_basic': {
        'env': 'Acrobot-v1',
        'arch': [64, 64, 64],
        'gamma': 0.99,
        'lr':5e-3,
        'num_steps': 500*6000,
        'num_episodes': 500*20,
        'max_episode_steps': 3000, 
        'schedule_timesteps': 3000*100,
        'initial_p': 1.0,
        'final_p': 0.1,
        'learning_freq': 1,
        'target_update_freq': 3000*30,
        'replay_buffer_size': 3000*20,
        'batch_size': 50,
        'augmented_reward': None,
        'save_freq': 3000*20,
    },
    'CP_basic': {
        'env': 'CartPole-v0',
        'arch': [64],
        'gamma': 0.99,
        'lr':5e-4,
        'num_steps': 500*200,
        'num_episodes': 500*4,
        'schedule_timesteps': 500*20,
        'initial_p': 1.0,
        'final_p': 0.01,
        'learning_freq': 1, # critical parameter
        'target_update_freq': 500*2,
        'replay_buffer_size': 500*100,
        'batch_size': 32,
        'augmented_reward': None,
        'save_freq': 500*20,
    },
	'LL_basic': {
        'env': 'LunarLander-v2',
        'arch': [64],
        'gamma': 0.99,
        'lr':5e-4,
        'num_steps': 500*200,
        'num_episodes': 1000*10,
        'max_episode_steps': None,
        'schedule_timesteps': 500*20,
        'initial_p': 1.0,
        'final_p': 0.01,
        'learning_freq': 1, # critical parameter
        'target_update_freq': 500*2,
        'replay_buffer_size': 500*100,
        'batch_size': 32,
        'augmented_reward': None,
        'save_freq': 500*20,
    },
	#larger exploration (500*500)
    'LL_e500': {   
		'env': 'LunarLander-v2',
        'schedule_timesteps': 500*500, # 10% of approx. all steps within num_episodes
    },
	#larger Replaybuffer (250*500)
    'LL_rpb250': {
        'env': 'LunarLander-v2',
        'replay_buffer_size': 500*250
    },
	#larger replaybuffer (500*500)
    'LL_rpb500': {
        'env': 'LunarLander-v2',
        'replay_buffer_size': 500*500
    },
	#clip gradients (10)
    'LL_gc10': {
        'env': 'LunarLander-v2',
        'grad_clip': 10 ,
    },
	#clip gradients (5)
    'LL_gc5': {
        'env': 'LunarLander-v2',
        'grad_clip': 5 ,
    },
	#priotizes sampling (alpha 0.2)
    'LL_prio1': {
        'env': 'LunarLander-v2',
        'prioritized': True,
		'prioritized_alpha': 0.2,
		'prioritized_beta0': 0.5,
    },
	#priotizes sampling (alpha 0.8)
    'LL_prio2': {
        'env': 'LunarLander-v2',
        'prioritized': True,
		'prioritized_alpha': 0.8,
		'prioritized_beta0': 0.5,
    },
	#2nd layer (64-64)
    'LL_64_64': {
        'env': 'LunarLander-v2',
        'arch': [64, 64],
    },
	#2nd layer (64-32)
    'LL_64_32': {
        'env': 'LunarLander-v2',
        'arch': [64, 32],
    },
	#2nd layer (64-96)
    'LL_64_96': {
        'env': 'LunarLander-v2',
        'arch': [64, 96],
    },
	# train less (/50 steps)
    'LL_frqz50': {
        'env': 'LunarLander-v2',
        'learning_freq': 50,
    },
	# train less, update more (100 batches /50 steps)
    'LL_frqz50_smp100': {
        'env': 'LunarLander-v2',
        'learning_freq': 50,
		'num_samples': 100,
    },
	# update more (100 batches /1 steps)
    'LL_frqz1_smp100': {
        'env': 'LunarLander-v2',
		'num_samples': 100,
    },
    # extended exploration + larger rp buffer
    'LL_e500_rpb500': {
        'env': 'LunarLander-v2',
        'schedule_timesteps': 500*500,
        'replay_buffer_size' : 500*500
    },
    # heavy gradient clipping
    'LL_gc1': {
        'env': 'LunarLander-v2',
        'grad_clip': 1
    },
    # very minor clipping
    'LL_gc20': {
        'env': 'LunarLander-v2',
        'grad_clip': 20
    },

	#TODO Softmax TD learning
}
