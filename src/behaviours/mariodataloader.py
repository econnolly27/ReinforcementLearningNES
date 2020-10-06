import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform, color, img_as_float, img_as_int
from PIL import Image
from scipy import misc

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, utils

class DatasetMario(Dataset):
    def __init__(self, file_path, csv_name, transform_in=None, behaviour=None):
        self.data = pd.read_csv(file_path+"/"+csv_name)
        self._path = file_path+"/"
        self.transform_in = transform_in

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load state and image from Mario FCEUX (after a dataset has been created with the Lua script and FM2)
        image_file = os.path.abspath(self.data.iloc[index, 0]) #.values.astype(np.uint8).reshape((1, 28, 28))
        state_fn = self.data.iloc[index, 1]

        # Get image
        img_rgba = Image.open(image_file)
        image_data = img_rgba.convert('RGB')

        # Get data from state.json
        with open(state_fn) as data_file:
            data = data_file.read()
            state_data = json.loads(data)

        # Get controller 1 info
        if "controller" in state_data:
            # {'A', 'B', 'down', 'left', 'right', 'select', 'start', 'up': False}
            # up, down, left, right, A, B, start, select
            control_data = np.asarray(list(state_data['controller'][0].values()), dtype=np.int)
            # control_data = np.asarray(list(range(0,len(control_data)))) * control_data
        else:
            control_data = np.zeros(8, dtype=np.int)

        # Get mario info
        if "mario" in state_data:
            mario_handler = state_data['mario']
            mpos = mario_handler["pos"]
            x1,y1,x2,y2 = mpos[0]-1,mpos[1]-8,mpos[2],mpos[3]-8 # For all items, this offsets work
            mario_handler['pos'] = [x1,y1,x2,y2]
            mario_handler['pos'] = np.asarray(mario_handler['pos'], dtype=np.int).reshape((2, 2))

            coins = mario_handler['coins']
            level_type = mario_handler['level_type']
            lives = mario_handler['lives']
            pos1, pos2 = mario_handler['pos']
            state = mario_handler['state']
            world = mario_handler['world']
            level = mario_handler['level']
            tmp_array = [state, lives, world, level, level_type, coins, pos1[0], pos1[1], pos2[0], pos2[1]]
            mario_data = np.asarray(tmp_array)
        else:
            mario_data = np.zeros(9, dtype=np.int)

        # Get enemy info
        #enemy_data = []
        enemy_data = np.asarray([0], dtype=np.int64)
        for i in range(0,5):
            key = "enemy"+str(i)
            if key in state_data:
                enemy_data = np.asarray([1], dtype=np.int64)
                #enemy_data.append(np.asarray(state_data["enemy"+str(i)], dtype=np.int))
            # else:
                # enemy_data.append(None)

        if self.transform_in is not None:
            image_data = self.transform_in(image_data)

        sample = {'image': image_data, 'state': torch.from_numpy(control_data).type(torch.long),
                  'mario': torch.from_numpy(mario_data), 'enemy': enemy_data, 'ifile': self.data.iloc[index, 0], 'sfile': self.data.iloc[index, 1]}

        return sample


class DatasetMarioBx(Dataset):
    def __init__(self, file_path, csv_name, buttontrain, transform_in=None):
        self.data = pd.read_csv(file_path+"/"+csv_name)
        self._path = file_path+"/"
        self.transform_in = transform_in
        # None is added to capture those instances where no button has been pressed
        self._buttons = {'a': 0, 'b': 1, 'down': 2, 'left': 3, 'right': 4, 'select': 5, 'start': 6, 'up': 7, 'None': 8}
        self._button = buttontrain

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load state and image from Mario FCEUX (after a dataset has been created with the Lua script and FM2)
        image_file = self._path + self.data.iloc[index, 0]#.values.astype(np.uint8).reshape((1, 28, 28))
        state_fn = self._path + self.data.iloc[index, 1]
        label_button = self.data.iloc[index, 2]

        # Get image
        img_rgba = Image.open(image_file)
        image_data = img_rgba.convert('RGB')

        # Get data from state.json
        with open(state_fn) as data_file:
            data = data_file.read()
            state_data = json.loads(data)

        # 1 means press button!
        if label_button == self._button:
            control_data = np.asarray([1], dtype=np.int64)
        else: # everything else
            control_data = np.asarray([0], dtype=np.int64)

        if self.transform_in is not None:
            image_data = self.transform_in(image_data)

        sample = {'image': image_data, 'state': torch.from_numpy(control_data).type(torch.float32), 'frame': '0'}

        return sample


class DatasetMarioBxv2(Dataset):
    # Here the loader will return i and i+1, i+2, ..., n
    def __init__(self, csv_name, buttontrain, transform_in=None):
        self.data = pd.read_csv(csv_name) # pd.read_csv(file_path+"/"+csv_name)
        self._path = "./"+"/"
        self.transform_in = transform_in
        # None is added to capture those instances where no button has been pressed
        self._buttons = {'a': 0, 'b': 1, 'down': 2, 'left': 3, 'right': 4, 'select': 5, 'start': 6, 'up': 7, 'None': 8}
        self._button = buttontrain
        self._noframes = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images = []
        # Load state and image from Mario FCEUX (after a dataset has been created with the Lua script and FM2)
        self._noframes = self.data.iloc[index,0]
        seq = self.data.iloc[index, 1]
        files = self.data.iloc[index, 4].replace("'","").strip('][').split(', ')
        for ifiles in files:
            image_file = self._path + ifiles

            # Get image
            img_rgba = Image.open(image_file)
            image_data = img_rgba.convert('RGB')

            if self.transform_in is not None:
                image_data = self.transform_in(image_data)

            images.append(image_data)

        label_button = self.data.iloc[index, 6]  # label is from last image in the list above
        
        # CLASSIFICATION: 1 means press button!
        # if self._button in label_button:
        #     control_data = np.asarray([1], dtype=np.int64)
        # else: # everything else
        #     control_data = np.asarray([0], dtype=np.int64)

        # REGRESSION: Compute prob of pressing button in noFrames
        if type(label_button) != float:
            prob_ocurr = label_button.count(self._button) / self._noframes
            control_data = np.asarray([prob_ocurr], dtype=np.float)
        else: # everything else
            control_data = np.asarray([0.0], dtype=np.float)

        sample = {'seq': seq, 'image': images, 'state': torch.from_numpy(control_data).type(torch.float32)}
        return sample


if __name__ == "__main__":
    # ***********************
    # DEMO
    # ***********************

    transform2apply = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

    dataset = DatasetMarioBxv2(csv_name=os.path.abspath("./bx_data/gerardo120719_5f.csv"), buttontrain='b', transform_in=transform2apply)
    # dataset = DatasetMarioBx(file_path="./", csv_name="./bx_data/allitems_A.csv", buttontrain='a', transform_in=transform2apply)
    print(len(dataset))
    sample = dataset[15]
    print(sample)

    # mario_dataset = DatasetMario(file_path="./", csv_name="allitems.csv", transform_in=transform2apply)
    # sample = mario_dataset.__getitem__(0)
    # print(sample)

    