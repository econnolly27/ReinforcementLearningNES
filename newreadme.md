# Reinforcement Learning in NES Games
# Level 4 Individual Project
# Erin Connolly 2314064C

# Readme

Each game has its own directory containing the code to train and test models.'retro_integration' contains the Retro data for each game, for example the reward and game data. 'testing' contains the code used to extract data from each game run, generating and saving graphs. It also contains a random agent used for comparison. 

## Build instructions

### Requirements

It is recommended to run this project within a virtual environment. The requirements are:

* Python 3.7
* Packages: listed in `requirements.txt` 
* Cuda 11 from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork
* Tested on Ubuntu 20.10
* Requires 32GB of RAM and 8GB of VRAM 

### Build steps

This repository does not include NES ROM files needed to run the code. These can be found by searching online the hash found in the rom.sha file in each game's folder within the retro-integration directory. Once downloaded, place the ROM file in the game's retro-integration folder and rename it rom.nes. 

To run the code, from the `src` directory:

* Train model: `python game/algorithm/train.py` For example: `python mario/PPO/train.py --lr 1e-5 --num_global_steps 4e6`
* Test model: `python game/algorithm/train.py` For example: `python gradius/a3c/test.py --num_processes 2`


### Test steps

List steps needed to show your software works. This might be running a test suite, or just starting the program; but something that could be used to verify your code is working correctly.

Examples:

* Run automated tests by running `pytest`
* Start the software by running `bin/editor.exe` and opening the file `examples/example_01.bin`


### Note

Please note that training an agent is computationally intensive. On slower machines, it may help to reduce the number of processes in train.py to 2 from 4. 