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
DATASET = "MaximumCoins_ALL"
#{0: 'a', 1: 'b', 2: 'd', 3: 'l', 4: 'r', 7: 'u'}
output_button = 'a'
proportion_nonbutton = 0.8
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
        dataset_writer.writerow(['image', 'state', 'button'])
        for each in bx:
            dataset_writer.writerow([each['image'], each['state'], each['bx']])

def compose(bx, bxnone):
    
    bx_size = len(bx)
    # bx_indices = list(range(bx_size))
    bxnone_size = len(bxnone)
    # bxnone_indices = list(range(bxnone_size))
    
    random_seed = 42
    np.random.seed(random_seed)

    bx_concat = None
    if bx_size <= bxnone_size:
        #np.random.shuffle(bxnone)
        bxnone_random = bxnone[:int(bx_size*proportion_nonbutton)]
        bx_concat = bx + bxnone_random
    else:
        print("BX bigger!!!!!!!!!!!!!!!")
        #np.random.shuffle(bx)
        bx_random = bx[:int(bx_size*proportion_nonbutton)]
        bx_concat = bx_random + bxnone

    print("====> Total bx_concat: " + str(len(bx_concat)) + "; bx: " + str(len(bx)) + "; x2 bx: " + str(2*len(bx)) + "; bxnone: " + str(len(bxnone)))
    return bx_concat



# dataset = md.DatasetMario(file_path="./", csv_name=DATASET+".csv")
dataset = pd.read_csv("./"+bx_data_path+DATASET+".csv")
bxAll = []
bxRest = []
length_data = len(dataset)
print_every = length_data // 10
for i in range(0,len(dataset)):
    btn = dataset.iloc[i, 2]
    sample = {'seq': i, 'image': dataset.iloc[i, 0], 'state': dataset.iloc[i, 1], 'bx': dataset.iloc[i, 2]} 
    if output_button in btn:
        bxAll.append(sample)
    else:
        bxRest.append(sample)   
    if (i + 1) % (print_every + 1) == 0:
        print(sample)

print("Button: " + str(len(bxAll)))
print("Rest: " + str(len(bxRest)))
bxConcat = compose(bxAll, bxRest)
save_csv(bx_data_path, output_button, bxConcat)
print("Done!")
