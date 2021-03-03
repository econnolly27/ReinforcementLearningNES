# Reinforcement Learning in NES Games
# Level 4 Individual Project
# Erin Connolly 2314064C

## Setup Instructions

* Get Anaconda (version doesn't matter)
* Get CUDA 11 from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork
  * Make sure to download "deb-network" and follow the instructions in the page
* `conda create --name project python=3.7.5`
* Activate mariobm environment with: `conda activate project`
* Following commands should be run inside project environment:
  * `conda install pytorch torchvision cudatoolkit=11 -c pytorch`
  * `conda install scikit-learn scipy pandas matplotlib`
  * `conda install -c conda-forge scikit-image scikit-plot`
  * `conda install scipy scikit-image scikit-learn pillow pandas numpy matplotlib imageio`
  * `pip3 install opencv-python`
  * `pip3 install gym-retro==0.8.0` (If you get an error about there will be errors after 2020 while updating or installing packages, just ignore it)
## Extra notes

* Activate env: `conda activate project`
* Check GPU usage: `watch nvidia-smi`

