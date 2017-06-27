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

## Google Cloud Platform Deployment
This is a step-by-step guide of what I did in order to get deploy and run our docker container on google compute engine.

First you need to sign-up for their service. You will get $300 of free computing time with a max period of 12 months. It will ask for credit card details, but as long as you do not upgrade to a paid plan, you won't be charged.

Before we start you need to [create a project](https://console.cloud.google.com/cloud-resource-manager)

### Get the `gcloud` command line tools (ubuntu)
(Taken from their docs)
```bash
# Create an environment variable for the correct distribution
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"

# Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk

## Initilize your account settings
gcloud init

To continue, you must log in. Would you like to log in (Y/n)? Y

# Next select the project you've created beforehand.
Pick cloud project to use:
 [1] [my-project-1]
 [2] [my-project-2]
 ...
 Please enter your numeric choice:
 
# You probably want to choose europe-west1-b (there's info on why too choose which online)
 Which compute zone would you like to use as project default?
 [1] [asia-east1-a]
 [2] [asia-east1-b]
 ...
 [14] Do not use default zone
 Please enter your numeric choice:
 
 
 gcloud has now been configured!
 You can use [gcloud config] to change more gcloud settings.

 Your active configuration is: [default]
```

### Build and push your docker image
1. Enable billing in your project settings (this should be on by default if this was your first project)
2. [Enable the API](https://console.cloud.google.com/flows/enableapi?apiid=containerregistry.googleapis.com&redirect=https:%2F%2Fcloud.google.com%2Fcontainer-registry%2Fdocs%2Fquickstart)
3. Build the image
```bash
cd $REPO_ROOT
docker build -t IMAGE_NAME .
```
4. Tag the image
```
docker tag IMAGE_NAME HOST_ID/PROJECT_ID/IMAGE_NAME
```
HOST\_ID should probably be `eu.gcr.io` if you're in europe.  
PROJECT\_ID is the name you gave your project earlier.  
IMAGE_NAME is a name of your choosing.

5. push the image
```bash
gcloud docker -- push HOST_ID/PROJECT_ID/IMAGE_NAME
```

### Create a GCE VM instance
We're gonna create an instance with 4 vCPUs and ~15GBs RAM.

1. Google provides VM OSes which are optimized for running docker containers inside, you can get a list of current images like this:
```bash
gcloud compute images list \
  --project cos-cloud \
  --no-standard-images
  
NAME                     PROJECT    FAMILY      DEPRECATED  STATUS
cos-beta-60-9592-23-0    cos-cloud  cos-beta                READY
cos-dev-61-9655-0-0      cos-cloud  cos-dev                 READY
cos-stable-57-9202-74-0  cos-cloud                          READY
cos-stable-58-9334-74-0  cos-cloud  cos-stable              READY
cos-stable-59-9460-64-0  cos-cloud  cos-stable              READY
```
We're selecting the latest stable one: `cos-stable-59-9460-64-0`.

2. Create a single VM instance
```bash
gcloud compute instances create VM_NAME \
    --image IMAGE_NAME \
    --image-project cos-cloud \
    --zone ZONE \
    --machine-type TYPE_ID
```
We use  `cos-stable-59-9460-64-0` as `IMAGE_NAME` and `n1-standard-4` as `TYPE_ID`.
`VM_NAME` is of your choosing.

I created 12 VMs, 6 each in two zones `europe-west1-b` and `europe-west2-a` as trial users have a restriction of 24 cores per region (`europe-west1` defines the region, `-b` the zone) and named them `testbench-vm-1-0` until `testbench-vm-2-5`.

3. Connect to your vm
```bash
gcloud compute ssh VM_NAME \
    --project PROJECT_NAME \
    --zone ZONE
```
This might ask you to generate a private/public key pair and enter a password, or if you've already done that just might ask for just that password.

### Configure and run our docker image
Inside the VM you can now pull and run our docker image.
As you pushed your image to `gcr.io` which is a private repo, you need to setup some environment variables.
Fortunately, Google provides a script at `/usr/share/google/dockercfg_update.sh`.
```bash
/usr/share/google/dockercfg_update.sh && docker pull HOST_ID/PROJECT_ID/IMAGE_NAME
```

Finally, you're able to run your experiments.
```bash
mkdir -p ~/data && docker run --rm -v ~/data:/mnt/data HOST_ID/PROJECT_ID/IMAGE_NAME --config CONFIG --repeat COUNT
```
This will create a data directory in the VM\'s home directory and save the outputs of your experiment to it.
Your final setup could then look something like this ![Screenshot of 12 VM sessions](resources/screen.png)

### Automation
#### vCores and Tensorflow sessions
The number of vCores and the number of cores used by the TensorFlow session (set in `train.py`) seem to have a very huge impact on the overall performance of the system.
In the following test `Google Compute Engine VMs` have been used and setup up in an attempt to always utilize the newest Core chipsets (Skylake or Broadwell) available.
The provisioned vCores run, depending on the chipset, at around 2.0GHz - 2.6GHz. Memory size and speed was not investigated, however the smallest setup (1 vCore) does only come with 3.5GB of RAM, which we consider minimum (note that GCE VMs offer a setup `n1-highcpu-2` which only comes with 1.8GB RAM)
##### Number of Cores
Four configurations where tested: 1, 2, 4, and 8 cores.
Each ran a docker container containing an image with our project in it, which utilized exactly the number of cores for its TensorFlow session has its host offered (1, 2, 4, and 8, respectively). The container was handed off to the docker daemon and the output directed to a file `experiment.log` to not slow down the process by outputting onto the console. The testbench configuration used is `AB_e10_short_de` (no reason to choose one in particular).
Following are the results in terms of CPU usage and estimated times of the experiment to finish (as predicted by the script itself).
###### One core and one core used by TF session
![One core and one core used by TF session](resources/vcore1-tf1.jpg)
> ETA after 100 episodes: 9h34min

###### Two core and two core used by TF session
![Two cores and two cores used by TF session](resources/vcore2-tf2.jpg)
> ETA after 100 episodes: 11h03min

> As this result was very unexpected I repeated it on a new VM with very similar results (ETA: 11h14min).

###### Four core and four core used by TF session
![Four cores and four cores used by TF session](resources/vcore4-tf4.jpg)
> ETA after 100 episodes: 6h15min

###### Eight core and Eight core used by TF session
![Eight cores and Eight cores used by TF session](resources/vcore8-tf8.jpg)
> ETA after 100 episodes: 4h19min

#### Cores allocated by Tensorflow
In an attempt to increase CPU usage when more than one core is present, an experiment has been conducted with two virtual cores on the host, while tensorflow attempts to allocate four.
![Two cores and Four cores used by TF session over six hours](resources/vcore2-tf4-6h.jpg)
> Very shaky behaviour with a clear declining trend

![Two cores and Four cores used by TF session over twelve hours](resources/vcore2-tf4-12h.jpg)
> It seems at first when the experiment started (shortly after 9AM), everything went well...until it didn't.

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
