# Reinforcement Learning in NES Games
# Level 4 Individual Project
# Erin-Louise Connolly 2314064C

# Readme

Each game has its own directory containing the code to train and test models. `retro_integration` contains the Retro data for each game, for example the reward and game data. `testing` contains the code used to extract data from each game run, generating and saving graphs.

## Build instructions

### Requirements

It is recommended to run this project within a virtual environment. The requirements are:

* Python 3.7.5
* Tested on Ubuntu 20.10
* Requires 32GB of RAM and 8GB of VRAM 
* Installation instructions are in `manual.md`. 

### Build steps

This repository does not include NES ROM files needed to run the code. These can be found by searching online the hash found in the rom.sha file in each game's folder within the retro-integration directory. Once downloaded, place the ROM file in the game's retro-integration folder and rename it rom.nes. 

To run the code, from the `src` directory:

* Train model: `python game/algorithm/train.py` For example: `python mario/PPO/train.py --lr 1e-5 --num_global_steps 4e6`
* Test model: `python game/algorithm/train.py` For example: `python gradius/a3c/test.py --num_processes 2`
    *  Ensure there is a saved model located within the trained_models folder for that game, with the naming scheme `a3c_game` or `ppo_game` e.g. `a3c_gradius`. 

### Test steps

Test training a model: `python mario/PPO/train.py`

Test testing a model: `python mario/PPO/train.py` An example model is provided for testing purposes. 


The training or testing process can be stopped by using Ctrl-C in the command line it is running in.

### Note

Please note that training an agent is computationally intensive. On slower machines, it may help to reduce the number of processes in train.py to 2 from 4. 
