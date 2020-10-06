
import json
import os
import sys

import matplotlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from behaviours.behaviour import Behaviour, trainNetBx
from behaviours.feature_extraction import FeatureExtraction
from behaviours.gencsv_behaviour import GenerateDatasetBx
from behaviours.mariodataloader import DatasetMarioBxv2
from behaviours.utils import create_datasets_split

matplotlib.use('Agg')



def load_experiment(filename):
    with open(filename) as json_file:
        opt = json.load(json_file)

    opt['feat_path'] = os.path.abspath(opt['feat_path'])
    opt['data_path'] = os.path.abspath(opt['data_path'])
    opt['bx_data_path'] = os.path.abspath(opt['bx_data_path'])
    return opt


if __name__ == "__main__":

    opt = load_experiment("./exp3.json")
    
    # Generate dataset?
    if os.path.isfile(opt["bx_data_path"]):
        print("Using existing file: "+opt["bx_data_path"])
    else:
        gendb = GenerateDatasetBx(opt["dataset"], opt["data_path"], opt["no_frames"], opt["bx_data_path"])
        gendb.run()

    # Setting up PyTorch stuff
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (opt['use_cuda'] and torch.cuda.is_available()) else "cpu")

    # Feature extraction
    feat_extract = FeatureExtraction(device)
    feat_extract.load_state_dict(torch.load(opt['feat_path']))
    feat_extract.to(device)

    transform2apply = transforms.Compose([transforms.Resize((opt['img_size'],opt['img_size'])), transforms.ToTensor()])
                                        
    # ********************************
    # Train Behaviours!
    for eachBx in opt["buttons"]:
        dataset = DatasetMarioBxv2(csv_name=opt["bx_data_path"], buttontrain=eachBx, transform_in=transform2apply)    
        train_loader, validation_loader = create_datasets_split(dataset, opt['shuffle_dataset'], opt['validation_split'], opt['batch_train'], opt['batch_validation'])
        
        bxmodel = Behaviour(opt["no_frames"]).to(device)
        trainNetBx(bxmodel, feat_extract, train_loader, validation_loader, n_epochs=opt['epochs'], lr=opt['learning_rate'], device=device, namebx="bx"+eachBx)
        print("Training "+eachBx+" done!")
    # ********************************

    print("Done!")
