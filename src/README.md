# mario-bm

## Original data

* https://www.dropbox.com/s/prae3xhfp18i90n/data.zip?dl=0

## Setup Instructions

* Get Anaconda (version doesn't matter)
* Get CUDA 10.1 from https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork
  * Make sure to download "deb-network" and follow the instructions in the page
* `conda create --name mariobm python=3.7.5`
* Activate mariobm environment with: `conda activate mariobm`
* Following commands should be run inside mariobm environment:
  * `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
  * `conda install scikit-learn scipy pandas matplotlib`
  * `conda install -c conda-forge scikit-image scikit-plot`
  * `conda install scipy scikit-image scikit-learn pillow pandas numpy matplotlib imageio`
  * `pip3 install opencv-python`
  * `pip3 install gym-retro==0.8.0` (If you get an error about there will be errors after 2020 while updating or installing packages, just ignore it)

## Training the AutoEncoder and Behaviours (current version)

1. After downloading the above date, uncompress it and copy folders into the `data` folder. See README inside folder
2. Now, in behaviours folder, run `gencsv_fe.py`. `all_items` is the default dataset, you can change the dataset to be used using the folder names (see point above).To do this, open `gencsv_fe.py` and change the `DATASET` variable. Run this file from `mario-bm` folder (parent folder), e.g. `python behaviours/gencsv_fe.py`; or using VS Code debugger, e.g. F5 in the file.
3. Run `feature_extraction.py` from `behaviours` folder to train the autoencoder (inside the file, you can change the name of the dataset, line 223).
4. After the autoencoder is trained, aka the feature extraction network, you can train the behaviours network using `mario_bx.py`. The behaviours network is defined in `behaviours/behaviour.py`. `mario_bx.py` just calls needed files and should be run from `mario-bm` or using VS Code. Remember to change the experiment file name in Line 35 in `mario_bx.py`. NOTE: At the moment, there are two experiments defined, `test.json` and `exp3.json`. In the JSON experiment file, you can define how many frames the dataset generation should skip. This trick helps the network to not overfit the data (need futher experimentation though)

## For BBRL: Experiments -- Gerardo Notes

* Run 3 experiments consisting of:
  * 5 frames (exp1 in results)
  * 4 frames (exp2 in results)
  * 3 frames (exp3 in results)

* Seperate behaviours structure between feature extraction and action. Follow Ameya's training scheme

## Extra notes

* Create environment.yml: `conda env export > environment.yml`
* Create env from file: `conda env create -f environment.yml`
* Activate env: `conda activate mariobm`
* ROMS: http://nesninja.com/public/GoodNES_3.14_goodmerged/nes/