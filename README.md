# Deep Reinforcment Learning Project 2017
[Deep RL Project Calendar](https://calendar.google.com/calendar/ical/fuc8n5c750pte22c32kgi940ug%40group.calendar.google.com/private-42a8e26f0d1130a135a2a1fd08accb6a/basic.ics)

## Project Plan

- [] Finish Double Pendulum environment (did we miss [this](https://gym.openai.com/evaluations/eval_NCtq2gxEQYZ78yvTutpQw)?)
    - save samples to array (state, action, successive state, reward)
      discretization via size of dt (size of dt has drastic effect on learning)
    - add friction
    - apply torque: w\_i+1 = w\_i + F/m * dt (only on first joint)
    - define reward function (maybe punish high forces)
    - add a wrapper class to access environment
    
- [] Testing
    - Episodes
    - Score (Quality of strategy)
    - Note: Set 2nd mass of pendulum to zero to mimic normal inverted pendulum
    - compare different reward functions
    
- [] Add/Change architecture elements
    - hidden layers
    - convolutional layers
    
- Ideas
    - [] number of hidden layer
	- [] rectifier units instead of hyperbolic tangent (!!! Xavier initialization is for sigmoidal units only I think)
	- [] buffer size (experience replay)
	- [] discount factor gamma
        WENDELIN's comment: 
		>too large = unstable, too small = doesn't transport reward far enough
		>there is some meassure of how robust the network is, by changing gamma and calculate number of steps needed (didn't understand that too well)
	- [] Optimizer hyperparameters (Adam / RMSprop)
				
- BONUS: (if time)
    - [] Prioritized replay (hint to goal heuristic), like sample 10%:
	    ( close to goal, rest-> uniform))
		( sample more where large TD-error and maybe add annealing sampling from there)
	  	( sample often near the goal in beginning and anneal to uniform or large TD-error )				
	- [] replace ringbuffer via uniform sampling instead of start->end->start..
	- [] double networks?? 
	- [] duelling networks (let's see if that even makes sense for our games)    

## Paper Guidelines
Wendilin is ok to only receive a single paper, needs to have independently written parts with clear assignment.

Wendilin's writing sequence:
* preliminary
* method
* experiment
* introduction (including results)
				
## Misc. Notes
### OpenAI gym installation
IF pachi-py fails to build wheel:
use `MACOSX_DEPLOYMENT_TARGET=10.12 pip install -e .[all]` (for bash/zsh)
or `env MACOSX_DEPLOYMENT_TARGET=10.12 pip install -e .[all] (for fish)`
... installer seems to assume OSX 10.5 as build env by default.
