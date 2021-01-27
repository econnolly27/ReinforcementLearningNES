# Reinforcement Learning in NES Games
# Level 4 Individual Project
# Erin Connolly 2314064C

## Setup Instructions

* Get Anaconda (version doesn't matter)
* Get CUDA 11 from https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork
  * Make sure to download "deb-network" and follow the instructions in the page
* `conda create --name mariobm python=3.7.5`
* Activate mariobm environment with: `conda activate mariobm`
* Following commands should be run inside mariobm environment:
  * `conda install pytorch torchvision cudatoolkit=11 -c pytorch`
  * `conda install scikit-learn scipy pandas matplotlib`
  * `conda install -c conda-forge scikit-image scikit-plot`
  * `conda install scipy scikit-image scikit-learn pillow pandas numpy matplotlib imageio`
  * `pip3 install opencv-python`
  * `pip3 install gym-retro==0.8.0` (If you get an error about there will be errors after 2020 while updating or installing packages, just ignore i
  * `pip install tensorboardx`
## Extra notes

* Create environment.yml: `conda env export > environment.yml`
* Create env from file: `conda env create -f environment.yml`
* Activate env: `conda activate mariobm`
* ROMS: http://nesninja.com/public/GoodNES_3.14_goodmerged/nes/
* Check GPU usage: `watch nvidia-smi`

