# mario-bm

120719 => I think world 3
120719 => 4/2

* Create environment.yml: `conda env export > environment.yml`
* Create env from file: `conda env create -f environment.yml`
* Activate env: `conda activate mariobm`

## Original data

* https://www.dropbox.com/s/prae3xhfp18i90n/data.zip?dl=0

## Package Install

* conda create --name mariobm python=3.7.5
* conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
* conda install scikit-learn scipy pandas matplotlib
* conda install -c conda-forge scikit-image scikit-plot
* conda install scipy scikit-image scikit-learn pillow pandas numpy matplotlib imageio
* pip install stable-baselines3

### For gym-retro, install

* sudo apt-get install zlib1g-dev
* After cloning above repo, do pip install -e . (gym-retro is included in this repo for version control purposes, so no need to clone)

ROMS: http://nesninja.com/public/GoodNES_3.14_goodmerged/nes/

## For BBRL: Training the AutoEncoder

1. After downloading the above date, uncompress it and copy folders into the `data` folder. See README inside folder
2. Now run `generate_csv.py`. By default, `all_items` is the default dataset, you can change the dataset to be used using the folder names (see point above). To do this, open `generate_csv.py` and change the `DATASET` variable. 
3. Run `feature_extractionv5.py` to train the autoencoder (inside the file, you can change the name of the dataset, look for `all_items`)

## For BBRL: Training Behaviours

1. XXX

## For BBRL: Experiments

* Run 3 experiments consisting of:
  * 5 frames (exp1 in results)
  * 4 frames (exp2 in results)
  * 3 frames (exp3 in results)

* Seperate behaviours structure between feature extraction and action. Follow Ameya's training scheme
