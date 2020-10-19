# Implements a data generation class for training behaviours (behaviours.py). This is called
# inside mario_bx.py and should not be run as standalone. File generated should go in "bx_data"
# Autho: Gerardo Aragon-Camarasa, 2020

import csv
from behaviours import mariodataloader as md
import numpy as np


class GenerateDatasetBx(object):
    def __init__(self, dataset, data_path, noFrames, bx_output):
        self.dataset = dataset
        self.noFrames = noFrames
        self.bx_output = bx_output
        self.data_path = data_path

    @staticmethod
    def retrieve_button(state_idx):
        # buttons = {'a': 0, 'b': 1, 'down': 2, 'left': 3, 'right': 4, 'select': 5,
        # 'start': 6, 'up': 7, 'None': 8}
        buttons = {0: 'a', 1: 'b', 2: 'd', 3: 'l', 4: 'r', 7: 'u'}
        btn = ""

        if state_idx.size == 0:
            btn += ''
        for idx in state_idx:
            if idx == 5 or idx == 6:
                btn += ''
            else:
                btn += buttons[idx]

        return btn

    def save_csv(self, bx):
        with open(self.bx_output, mode='w', newline='', encoding='utf-8') as dataset_file:
            dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            dataset_writer.writerow(['noframe', 'seq', 'world', 'level', 'image', 'state', 'button'])
            for each in bx:
                dataset_writer.writerow([each['noframes'], each['seq'], each['world'], each['level'], each['image'], each['state'], each['bx']])

    def run(self):
        dataset = md.DatasetMario(file_path=self.data_path, csv_name=self.dataset+".csv")
        bxAll = []
        length_data = len(dataset)
        wrd = 0
        lvl = 0
        bx_list = ''
        print("World -> " + str(wrd))
        for i in range(self.noFrames-1, length_data):
            ifile = []
            for j in range(i+1-self.noFrames, i+1):
                ifile.append(dataset[j]['ifile'])

                s = dataset[j]['state'].numpy()  # state corresponds to the last image in the list!
                state_idx = np.where(s)[0]
                bx = self.retrieve_button(state_idx)
                bx_list += bx

            world = dataset[i]['mario'][2].item()
            level = dataset[i]['mario'][3].item()

            if world != wrd:
                print("World -> " + str(world))
                wrd = world

            if level != lvl:
                print("======> Level -> " + str(level))
                lvl = level

            sample = {'noframes': self.noFrames, 'seq': i, 'world': world, 'level': level, 'image': ifile, 'state': dataset[i]['sfile'], 'bx': bx_list}
            bxAll.append(sample)
            bx_list = ''

        print("All: " + str(len(bxAll)))
        self.save_csv(bxAll)
        print("Done!")
