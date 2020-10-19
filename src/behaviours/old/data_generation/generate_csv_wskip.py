import csv
import os

# ****************** 
# Only edit variable below with the name of the dataset
# This script should be placed at the same level as the "data" directory
# ******************
DATASET = "allitems"
skip = 1
# ******************
# ******************

# ******************
# DO NOT EDIT FROM HERE!!!
# ******************
data_path = "./data/"
file_path = data_path + DATASET + "/" + DATASET + "_wskip.csv"
with open(DATASET + "_wskip.csv", mode='w', newline='', encoding='utf-8') as dataset_file:
    dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    dataset_writer.writerow(['image', 'state'])

    base_names = []
    directory = os.fsencode(data_path + DATASET + "/images")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            #print(filename[:-4])
            base_names.append(filename[:-4])
            continue
        else:
            continue

    base_names.sort(key=int)
    i = 0
    for name in base_names:
        if i < 1:
            i += 1
            img_path = data_path + DATASET + "/images/" + name + ".png"
            state_path = data_path + DATASET + "/states/" + name + "_state.json"
            dataset_writer.writerow([img_path, state_path])
        else:
            i = 0

print("Done!")
