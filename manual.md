# User manual 

It is recommended to run this code inside a virual environment running Python 3.7.5.

* Get CUDA 11 from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork
  * Make sure to download "deb-network" and follow the instructions in the page
* Following commands should be run inside project environment:
  * `conda install pytorch torchvision cudatoolkit=11 -c pytorch`
  * `conda install scikit-learn scipy pandas matplotlib`
  * `conda install -c conda-forge scikit-image scikit-plot`
  * `conda install scipy scikit-image scikit-learn pillow pandas numpy matplotlib imageio`
  * `pip3 install opencv-python`
  * `pip3 install gym-retro==0.8.0`

This repository does not include NES ROM files needed to run the code. These can be found by searching online the hash found in the rom.sha file in each game's folder within the retro-integration directory. Once downloaded, place the ROM file in the game's retro-integration folder and rename it rom.nes. 

You can then train or test a model by running the code from the `src` directory:

* Train model: `python game/algorithm/train.py` For example: `python mario/PPO/train.py --lr 1e-5 --num_global_steps 4e6`
* Test model: `python game/algorithm/train.py` For example: `python gradius/a3c/test.py --num_processes 2`
   *  Ensure there is a saved model located within the trained_models folder for that game. 