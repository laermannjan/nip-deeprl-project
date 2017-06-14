# Deep Reinforcment Learning Project 2017
[Deep RL Project Calendar](https://calendar.google.com/calendar/ical/fuc8n5c750pte22c32kgi940ug%40group.calendar.google.com/private-42a8e26f0d1130a135a2a1fd08accb6a/basic.ics)

## Useful findings
* Video rendering: If `env.render()` is not called, a video will be produces according to the default schedule (from monitoring.py):

```python
def capped_cubic_video_schedule(episode_id):  
    if episode_id < 1000:  
        return int(round(episode_id ** (1. / 3))) ** 3 == episode_id  
    else:  
        return episode_id % 1000 == 0  
```
Should be resolved by calling `env.render()` per episode.
## Task Assi800gnments
- Seonguk  ###
    * --- 
- Jan 
    * Run testbench for LunarLander over multiple architectures : (100-100-100) vs (150-100-50)  a 3x times = 6 runs 
- Robert
    * Run testbench for LunarLander over multiple exploration rates : (10 000) vs (400 000)  a 3x times = 6 runs 
- Manuel
    * Run testbench for LunarLander over multiple architectures : (150-150-150) vs (200-150-100)  a 3x times = 6 runs 
- SHOULD TAKE ABOUT 30 MINS per run
    
## Project Plan


- timeline
    - [x] run vanilla DQN (statespace)
	- [x] find stable parameters (Cartpole,Acrobot,Lunarlander)
    - [x] move from vanilla DQN to baseline implementation 
    	- [X] find stable parameters in statespace (Acrobot,Lunarlander)
	- [x] adapt to imagespace 
	- [ ] find stable parameters in imagespace (Acrobot,Lunarlander)

- Notes
    - Change architecture
	- [x] rectifier units instead of hyperbolic tangent (!!! Xavier initialization is for sigmoidal units, LeCun init for ReLU
	- [x] discount factor gamma
        WENDELIN's comment: 
		>too large = unstable, too small = doesn't transport reward far enough
		>there is some meassure of how robust the network is, by changing gamma and calculate number of steps needed (didn't understand that too well)
				
- Ideas: (if time)
    	- [ ] Prioritized replay (hint to goal heuristic), like sample 10%:
	    ( close to goal, rest-> uniform))
		( sample more where large TD-error and maybe add annealing sampling from there)
	  	( sample often near the goal in beginning and anneal to uniform or large TD-error )				
	- [ ] replace ringbuffer via uniform sampling instead of start->end->start..

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
