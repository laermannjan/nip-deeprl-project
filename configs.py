default_conv_arch = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]

Configs = {
    'AB_basic': {
        'env': 'Acrobot-v1',
	    'arch': [64],
       	'gamma': 0.99,
        'lr':5e-4,
        'num_steps': 500*200,
        'num_episodes': 500*20,
        'max_episode_steps': 500,
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
    # smaller minibatch
    'LL_mb16': {
        'env': 'LunarLander-v2',
        'batch_size': 16
    },
    # larger minibatch
    'LL_mb128': {
        'env': 'LunarLander-v2',
        'batch_size': 128
    },
    'LL_256_128': {
        'env': 'LunarLander-v2',
        'arch': [256, 128]
    },
    'LL_256': {
        'env': 'LunarLander-v2',
        'arch': [256]
    },
    'LL_256_256': {
        'env': 'LunarLander-v2',
        'arch': [256, 256]
    },
    'LL_256_256_256': {
        'env': 'LunarLander-v2',
        'arch': [256, 256, 256]
    },
    'LL_256_256_512': {
        'env': 'LunarLander-v2',
        'arch': [256, 256, 512]
    },
    'LL_256_512': {
        'env': 'LunarLander-v2',
        'arch': [256, 512]
    },
    'LL_512_512': {
        'env': 'LunarLander-v2',
        'arch': [512, 512]
    },
    'LL_128_256_512': {
        'env': 'LunarLander-v2',
        'arch': [128, 256, 512]
    },
    'LL_512_256_128': {
        'env': 'LunarLander-v2',
        'arch': [512, 256, 128]
    },
    'LL_512_512_e10k': {
        'env': 'LunarLander-v2',
        'arch': [256, 256],
        'schedule_timesteps': 10000*200,
        'num_episodes': 1000*100,
        'lr': 1e-4,
        'batch_size': 64,
    },
    'LL_256_256_e10k': {
        'env': 'LunarLander-v2',
        'arch': [256, 256],
        'schedule_timesteps': 10000*200,
        'num_episodes': 1000*100,
        'lr': 1e-4,
        'batch_size': 64,
    },
    'LL_256_256_e10k_lf10': {
        'env': 'LunarLander-v2',
        'arch': [256, 256],
        'schedule_timesteps': 10000*200,
        'num_episodes': 1000*100,
        'lr': 1e-4,
        'batch_size': 64,
        'learning_freq': 10,
    },
    'LL_256_256_e10k_numsamples10': {
        'env': 'LunarLander-v2',
        'arch': [256, 256],
        'schedule_timesteps': 10000*200,
        'num_episodes': 1000*100,
        'lr': 1e-4,
        'batch_size': 64,
        'num_samples': 20,
    },
	# search a stable AB basic
    'AB_64': {
        'env': 'Acrobot-v1',
	    'num_episodes': 500*4,
		'arch': [64],
    },
	# search a stable AB basic
    'AB_64-64-1000': {
        'env': 'Acrobot-v1',
		'max_episode_steps': 1000,
	    'num_episodes': 500*4,

    },
	# search a stable AB basic
    'AB_64-1000': {
        'env': 'Acrobot-v1',
		'arch': [64],
	    'num_episodes': 500*4,
		'max_episode_steps': 1000,
    },
	##############################################################################
	# search a stable AB basic
    'AB_64': {
        'env': 'Acrobot-v1',
	    'num_episodes': 500*4,
		'arch': [64],
    },
	# search a stable AB basic
    'AB_64-64-1000': {
        'env': 'Acrobot-v1',
		'max_episode_steps': 1000,
	    'num_episodes': 500*4,

    },
	# search a stable AB basic
    'AB_64-1000': {
        'env': 'Acrobot-v1',
		'arch': [64],
	    'num_episodes': 500*4,
		'max_episode_steps': 1000,
    },

    #larger exploration (500*500)
      'AB_e500': {
      'env': 'Acrobot-v1',
          'schedule_timesteps': 500*1000, # 10% of approx. all steps within num_episodes, assume 500 steps per episode, prev. 500*500
      },
    #larger Replaybuffer (350*500)
      'AB_rpb300': {
          'env': 'Acrobot-v1',
          'replay_buffer_size': 500*600
      },
    #larger replaybuffer (600*500)
      'AB_rpb600': {
          'env': 'Acrobot-v1',
          'replay_buffer_size': 1000*600
      },
    #clip gradients (10) --optional
      'AB_gc10': {
          'env': 'Acrobot-v1',
          'grad_clip': 10 ,
      },
    #clip gradients (5) --optional
      'AB_gc5': {
          'env': 'Acrobot-v1',
          'grad_clip': 5 ,
      },
    #priotizes sampling (alpha 0.2)
      'AB_prio1': {
          'env': 'Acrobot-v1',
          'prioritized': True,
      'prioritized_alpha': 0.1,
      'prioritized_beta0': 0.5,
      },
    #priotizes sampling (alpha 0.8)
      'AB_prio2': {
          'env': 'Acrobot-v1',
          'prioritized': True,
      'prioritized_alpha': 0.9,
      'prioritized_beta0': 0.5,
      },
    #one smaller layer (32)
      'AB_32': {
          'env': 'Acrobot-v1',
          'arch': [32],
      },
    #one bigger layer (128)
      'AB_128': {
          'env': 'Acrobot-v1',
          'arch': [128],
      },
    #2nd layer (64-64)
      'AB_64_64': {
          'env': 'Acrobot-v1',
          'arch': [64, 64],
      },
    #2nd layer (64-32)
      'AB_64_32': {
          'env': 'Acrobot-v1',
          'arch': [64, 32],
      },
    #2nd layer (64-96)
      'AB_64_96': {
          'env': 'Acrobot-v1',
          'arch': [64, 96],
      },

      # extended exploration + larger rp buffer
      'AB_e500_rpb300': {
          'env': 'Acrobot-v1',
          'schedule_timesteps': 500*1000,
          'replay_buffer_size' : 500*600
      },
      # heavy gradient clipping
      'AB_gc1': {
          'env': 'Acrobot-v1',
          'grad_clip': 1
      },
      # very minor clipping
      'AB_gc20': {
          'env': 'Acrobot-v1',
          'grad_clip': 20
      },
    'LL_img_basic_test0': {
        'env': 'LunarLander-v2',
        'image': True,
        'arch': [256],
        'conv_arch': default_conv_arch,
        'schedule_timesteps': 500*100,
    },
    'LL_img_basic_test0_prio': {
        'env': 'LunarLander-v2',
        'image': True,
        'arch': [256],
        'conv_arch': default_conv_arch,
        'schedule_timesteps': 500*100,
        'prioritized': True,
    },
    'LL_img_basic_test1': {
        'env': 'LunarLander-v2',
        'image': True,
        'arch': [265, 64],
        'schedule_timesteps': 500*100,
        'conv_arch': default_conv_arch
    },
    'LL_img_basic_test2': {
        'env': 'LunarLander-v2',
        'image': True,
        'arch': [265, 128],
        'schedule_timesteps': 500*100,
        'conv_arch': default_conv_arch
    },
	#TODO Softmax policy improvement algorithm
	# pi = exp(beta*Q(s,a)) / \sum_{over a}(exp(beta*Q(s,a))
	# wobei beta \in [0, inf)
	# beta = 0 yields equal distribution over actions
	# lim beta to inf converges to taking the action with largest (Q(s,a))
	# try something like 0.2 und 2
}
