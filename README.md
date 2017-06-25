# Deep Reinforcment Learning Project 2017
[Deep RL Project Calendar](https://calendar.google.com/calendar/ical/fuc8n5c750pte22c32kgi940ug%40group.calendar.google.com/private-42a8e26f0d1130a135a2a1fd08accb6a/basic.ics)

## Setup
**Optional**: This project has been packaged as a python pip package (but was not published to PyPI). This makes it possible to run setup.py and use our main facilities (train.py, dual_monitor.py, utils.py, etc.) in other projects.

### Docker
#### Install docker (from [the docker docs](https://docs.docker.com/engine/installation/linux/ubuntu/#install-docker())):
1. 
```bash
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
```

2. 
```bash
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

3. Verify that the key fingerprint is `9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88`.
```
sudo apt-key fingerprint 0EBFCD88

pub   4096R/0EBFCD88 2017-02-22
      Key fingerprint = 9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid                  Docker Release (CE deb) <docker@docker.com>
sub   4096R/F273FCD8 2017-02-22
```

4. 
```bash
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

5. 
```bash
$ sudo apt-get update
$ sudo apt-get install docker-ce
```

6. Verify everything works correctly
```bash
$ sudo docker run hello-world
```

#### Build docker image and run the container
1. Navigate to the root directory of your clone of this repo. Then run:
```bash
$ cd docker
$ docker build -t laermannjan/nip-deeprl-docker:cpu .
```

2.Substitute `$REPO_ROOT` by the root directory to you clone of the repo. 
##### (Recommended) Use docker container as a service.
```bash
$ docker run --rm -itv $REPO_ROOT/data:/mnt/data laermannjan/nip-deeprl-docker:cpu --config config_name1
```
You can pass the same argumemts you would pass onto testbench.py to the docker container, like we did with `--config config_name1` in the example above. **Important:** make sure you do not change the mount point `/code` and that your repo is named `nip-deeprl-project`.
##### Interact with the container
```bash
$ docker run --rm -itv $REPO_ROOT/data:/mnt/data laermannjan/nip-deeprl-docker:cpu /bin/bash
```
This will run the image in a new container and open up an interactive bash entrypoint for you.
You will be able to access the code of this repo from inside that container at `/code/nip-deeprl-project`.
Inside you need to switch to custom conda environment:
```bash
$ source activate py35
```
Your command line should now look something like this: `(py35) root@382c92f920a0:~#`.

3. (Not yet implemented) You can also run a jupyter notebook instance as an entrypoint via: 
```bash
docker run --rm -itv  $REPO_ROOT/data:/mnt/data -p 8888:8888 laermannjan/nip-deeprl-docker:cpu
```
## SWIG\_Constant\_randInt Fix
From [Issue #17](https://github.com/laermannjan/nip-deeprl-project/issues/17)
Box2D needs swig3.0 to properly compile.
Ubuntu 14.10 and above should behave correctly. 14.04 Trusty Tahr does not have swig3.0 in default repositories, therefore we need to make backport repos available 
According to [Greg Brockman](https://github.com/openai/gym/issues/83#issuecomment-218357232) (these commands remove any installed older versions of Swig):

```bash
$ echo deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse | sudo tee /etc/apt/sources.list.d/box2d-py-swig.list
$ apt-get update && install -t trusty-backports swig3.0
$ apt-get remove swig swig2.0
$ ln -s /usr/bin/swig3.0 /usr/bin/swig
```

Additionally we can install a prebuilt binary of Box2D to eliminate another source of compile complications:
```bash
# Create conda env with python version 3.3, 3.4 or 3.5
$ conda create -n py35 python=3.5
$ source activate py35
# Install binaries
$ conda install -c https://conda.anaconda.org/kne pybox2d
```

## How to run experiments
The main logic is implemented inside the python package.
A complimentary test script `testbench.py` has been included outside of the package.
Additionally a complimentary `configs.py` has been included where you can define the setup for your experiments.

### Configs
You can pass a set of parameters for your experiment via the `--config CONFIG [CONFIG ...]` argument of `testbench.py`.
In this case `CONFIG` must be the name of a config defined in `conigs.py`, i.e. it must match a `key` in the `Configs` dict inside `configs.py`.

#### Example `configs.py`
(This is still an [open issue](https://github.com/laermannjan/nip-deeprl-project/issues/13))
```python
Configs = {
    'config_name1': {
        'key1': 'value1',
        'key2': value2
        ...
    },
    'config_name2': {
        'key1': 'value4',
        'key3': value3
        ...
    },
    ....
}
```

To match our guidelines on how to conduct experiments, `config.py`\'s `Configs` must contain three configs, namely `AB_basic`, `CP_basic` and `LL_basic`. These should define the baseline setting for each game respectively.

#### How to write a config
A config **must** define the `'env'` key and its value **must** be either `Acrobot-v1`, `Cartpole-v0` or `LunarLander-v1`.
Other than that you only need (and should) define the parameters you want changed with respect to the baseline config, e.g.:

```python
Configs = {
    'AB_basic': {
        'env': 'Acrobot-v1',
        'gamma': 0.99,
        'arch': [30,30,30]
    },
    'AB_test_gamma': {
        'env': 'Acrobot-v1'
        'gamma': 0.7
    }
        
}
```

#### Keys
All available keys have been defined in `testbench.py` as arguments of the command line, e.g.
```python
parser.add_argument("--env", type=str, choices=[spec.id for spec in envs.registry.all()], default="LunarLander-v2", help="name of the game")
```
An `argument` in `testbench.py` of type `"--argument-name"` gets translated into a `key` in `configs.py` of type `'argument_name'`, note the switch from hyphen to underscore.

### Running an experiment (only use if you do not use the docker container as a service)
A simple run can now be initiated by `python testbench.py --config AB_test_gamma`.
Checkout `testbench.py` for all possible arguments to pass, but note that if an argument is defined in the specified config it will take priority!
Command-line arguments can still be of use if you quickly want to change something for test purposes (but you should not use them for proper experiment evaluation).
Some noteworthy arguments are:
- `--repeat COUNT` - let\'s you specify how often an specific experiment should be repeated to minimise statistical errors.
- `--capture-videos` - as name suggests turns on video rendering, this will slow down the experiment.
- `--write-upon-reset` - write intermediary experiment results to fail after each episode. This makes everything more failsave but might slow down the experiment significantly depending on the data-storage connection.
- `--save-dir` - root of where you want your experiments outputs saved. If you pass a relative path, note that this is relative to from where you called the script, not where the script is located! *Note* per default this outputs to the `data` directory inside your clone of the repo. This makes it easy for you to share (i.e. push your result to github). **IMPORTANT** If you are inside docker, make sure that you either run the script from where it is located (i.e. `cd` to `/code/nip-deeprl-project` and run `python testbench.py ...` there) or specify `-save-dir` to be 
`/code/nip-deeprl-project/data`.


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
