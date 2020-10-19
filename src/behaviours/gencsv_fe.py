# Implements a data generation script for training an autoencoder (feature_extraction.py). This should
# be run as standalone, and file should go in "data"
# NOTE: Make sure to run this from the parent folder, i.e. from "mario-bm", run "python behaviours/gencsv_fe.py"  or using VS Code
# Autho: Gerardo Aragon-Camarasa, 2020

import csv
import os

# ****************** 
# Only edit variable below with the name of the dataset
# This script should be placed at the same level as the "data" directory
# ******************
DATASET = "gerardo110719"
# ******************
# ******************

# ******************
# DO NOT EDIT FROM HERE!!!
# ******************
data_path = os.path.abspath("./data/")
file_path = data_path + "/" + DATASET + ".csv"
with open(file_path, mode='w', newline='', encoding='utf-8') as dataset_file:
    dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    dataset_writer.writerow(['image', 'state'])

    base_names = []
    directory = os.fsencode(data_path + "/" + DATASET + "/images")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            #print(filename[:-4])
            base_names.append(filename[:-4])
            continue
        else:
            continue

    base_names.sort(key=int)
    for name in base_names:
        img_path = data_path + DATASET + "/images/" + name + ".png"
        state_path = data_path + DATASET + "/states/" + name + "_state.json"
        dataset_writer.writerow([img_path, state_path])

print("Done!")
