import csv
import os
import MarioData as md
import torch
import numpy as np
import pandas as pd

# ****************** 
# Only edit variable below with the name of the dataset
# This script should be placed at the same level as the "data" directory
# ******************
DATASET = "allitems_ALL"
world = [0] # list specifying the worlds to be included in the dataset
# ******************
# ******************

# ******************
# DO NOT EDIT FROM HERE!!!
# ******************
data_path = "./data/"
bx_data_path = "./bx_data/"

def save_csv(bx_data_path, namebx, bx):
    with open(bx_data_path + DATASET + "_" + namebx + ".csv", mode='w', newline='', encoding='utf-8') as dataset_file:
        dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataset_writer.writerow(['seq', 'world', 'level', 'image', 'state', 'button'])
        for each in bx:
            dataset_writer.writerow([each['seq'], each['world'], each['level'], each['image'], each['state'], each['bx']])

# def compose(bx, bxnone):
    
#     bx_size = len(bx)
#     # bx_indices = list(range(bx_size))
#     bxnone_size = len(bxnone)
#     # bxnone_indices = list(range(bxnone_size))
    
#     random_seed = 42
#     np.random.seed(random_seed)

#     bx_concat = None
#     if bx_size <= bxnone_size:
#         #np.random.shuffle(bxnone)
#         bxnone_random = bxnone[:int(bx_size*proportion_nonbutton)]
#         bx_concat = bx + bxnone_random
#     else:
#         print("BX bigger!!!!!!!!!!!!!!!")
#         #np.random.shuffle(bx)
#         bx_random = bx[:int(bx_size*proportion_nonbutton)]
#         bx_concat = bx_random + bxnone

#     print("====> Total bx_concat: " + str(len(bx_concat)) + "; bx: " + str(len(bx)) + "; x2 bx: " + str(2*len(bx)) + "; bxnone: " + str(len(bxnone)))
#     return bx_concat


dataset = pd.read_csv("./"+bx_data_path+DATASET+".csv")
bxAll = []
bxRest = []
length_data = len(dataset)
print_every = length_data // 10
for i in range(0,len(dataset)):
    sample = {'seq': dataset.iloc[i, 0], 'world': dataset.iloc[i, 1], 'level': dataset.iloc[i, 2], 'image': dataset.iloc[i, 3], 'state': dataset.iloc[i, 4], 'bx': dataset.iloc[i, 5]}
    if dataset.iloc[i, 1] in world:
        bxAll.append(sample)
    # else:
    #     bxRest.append(sample)   
    if (i + 1) % (print_every + 1) == 0:
        print(sample)

print("Size: " + str(len(bxAll)))
# print("Rest: " + str(len(bxRest)))
# bxConcat = compose(bxAll, bxRest)
str_world = ''.join([str(w) for w in world]) 
save_csv(bx_data_path, str_world, bxAll)
print("Done!")
