import csv
import os
import MarioData as md
import torch
import numpy as np

# ****************** 
# Only edit variable below with the name of the dataset
# This script should be placed at the same level as the "data" directory
# ******************
DATASET = "allitems"
# ******************
# ******************

# ******************
# DO NOT EDIT FROM HERE!!!
# ******************

def save_csv(bx_data_path, namebx, bx):
    with open(bx_data_path + DATASET + "_" + namebx + ".csv", mode='w', newline='', encoding='utf-8') as dataset_file:
        dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataset_writer.writerow(['image', 'state', 'button'])
        for each in bx:
            dataset_writer.writerow([each['image'], each['state'], each['bx']])

def compose_and_save(bx_data_path, bx_name, bx, bxnone):
    
    bx_size = len(bx)
    # bx_indices = list(range(bx_size))
    bxnone_size = len(bxnone)
    # bxnone_indices = list(range(bxnone_size))
    
    random_seed = 42
    np.random.seed(random_seed)

    bx_concat = None
    if bx_size <= bxnone_size:
        np.random.shuffle(bxnone)
        bxnone_random = bxnone[:bx_size]
        bx_concat = bx + bxnone_random
    else:
        print("BX bigger!!!!!!!!!!!!!!!")
        np.random.shuffle(bx)
        bx_random = bx[:bx_size]
        bx_concat = bx_random + bxnone

    print("====> Total bx_concat: " + str(len(bx_concat)) + "; bx: " + str(len(bx)) + "; x2 bx: " + str(2*len(bx)) + "; bxnone: " + str(len(bxnone)))

    save_csv(bx_data_path, bx_name, bx_concat)

dataset = md.DatasetMario(file_path="./", csv_name=DATASET+".csv")

bxUp = []
bxDown = []
bxLeft = []
bxRight = []
bxA = []
bxB = []
bxNone = []
bxStart = []
bxSelect = []

length_data = len(dataset)
print_every = length_data // 10
for i in range(0,length_data):
    d = dataset[i]
    s = d['state'].numpy()
    state_idx = np.where(s)[0]    
    if state_idx.size == 0:
        todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'None'}
        bxNone.append(todataset)
    else:
        # {0: 'A', 1: 'B', 2: 'down', 3: 'left', 4: 'right', 5: 'select', 6: 'start', 7: 'up'}
        for idx in state_idx:
            if idx == 7: #up
                todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'up'}
                bxUp.append(todataset)
            elif idx == 2: #down
                todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'down'}
                bxDown.append(todataset)

            if idx == 3: #left
                todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'left'}
                bxLeft.append(todataset)
            elif idx == 4: #right
                todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'right'}
                bxRight.append(todataset)

            if idx == 0: #A
                todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'a'}
                bxA.append(todataset)
            
            if idx == 1: #B
                todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'b'}
                bxB.append(todataset)
            
            if idx == 5:
                todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'select'}
                bxSelect.append(todataset)

            if idx == 6:
                todataset = {'seq': i, 'image': d['ifile'], 'state': d['sfile'], 'bx': 'start'}
                bxStart.append(todataset)
    
    if (i + 1) % (print_every + 1) == 0:
        print(todataset)

data_path = "./data/"
bx_data_path = "./bx_data/"

print("Up: " + str(len(bxUp)))
print("Down: " + str(len(bxDown)))
print("Left: " + str(len(bxLeft)))
print("Right: " + str(len(bxRight)))
print("A: " + str(len(bxA)))
print("B: " + str(len(bxB)))
print("Start: " + str(len(bxStart)))
print("Select: " + str(len(bxSelect)))
print("None: " + str(len(bxNone)))

print("Total: " + str(len(bxUp) + len(bxDown) + len(bxLeft) + len(bxRight) + len(bxA) + len(bxB) + len(bxStart) + len(bxSelect) + len(bxNone)))

compose_and_save(bx_data_path, "UP", bxUp, bxRight+bxDown+bxLeft+bxA+bxB+bxStart+bxSelect+bxNone)
compose_and_save(bx_data_path, "DOWN", bxDown, bxUp+bxRight+bxLeft+bxA+bxB+bxStart+bxSelect+bxNone)
compose_and_save(bx_data_path, "LEFT", bxLeft, bxUp+bxDown+bxRight+bxA+bxB+bxStart+bxSelect+bxNone)
compose_and_save(bx_data_path, "RIGHT", bxRight, bxUp+bxDown+bxLeft+bxA+bxB+bxStart+bxSelect+bxNone)
compose_and_save(bx_data_path, "A", bxA, bxUp+bxDown+bxLeft+bxRight+bxB+bxStart+bxSelect+bxNone)
compose_and_save(bx_data_path, "B", bxB, bxUp+bxDown+bxLeft+bxA+bxRight+bxStart+bxSelect+bxNone)

print("Done!")
