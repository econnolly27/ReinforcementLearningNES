import csv
import MarioData as md
import numpy as np

# ******************
# Only edit variable below with the name of the dataset
# This script should be placed at the same level as the "data" directory
# ******************
# "antipacifist" #'allitems' #"gerardo110719" # "MaximumCoins"
DATASET = "gerardo120719v2"
noFrames = 5  # no frame to look back in the past
# ******************
# ******************

# ******************
# DO NOT EDIT FROM HERE!!!
# ******************
data_path = "./data/"
bx_data_path = "./bx_data/"


def retrieve_button(state_idx):
    # buttons = {'a': 0, 'b': 1, 'down': 2, 'left': 3, 'right': 4, 'select': 5,
    # 'start': 6, 'up': 7, 'None': 8}
    buttons = {0: 'a', 1: 'b', 2: 'd', 3: 'l', 4: 'r', 7: 'u'}
    btn = ""

    if state_idx.size == 0:
        btn += 'x'
    for idx in state_idx:
        if idx == 5 or idx == 6:
            btn += 'x'
        else:
            btn += buttons[idx]

    return btn


def save_csv(bx_data_path, namebx, bx):
    with open(bx_data_path + DATASET + "_" + namebx + ".csv", mode='w', newline='', encoding='utf-8') as dataset_file:
        dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataset_writer.writerow(['noframe', 'seq', 'world', 'level', 'image', 'state', 'button'])
        for each in bx:
            dataset_writer.writerow([each['noframes'], each['seq'], each['world'], each['level'], each['image'], each['state'], each['bx']])


dataset = md.DatasetMario(file_path="./", csv_name=DATASET+".csv")
bxAll = []
bxRest = []
length_data = len(dataset)
print_every = length_data // 5
wrd = 0
lvl = 0
print("World -> " + str(wrd))
for i in range(noFrames-1, length_data):
    ifile = []
    for j in range(i+1-noFrames, i+1):
        ifile.append(dataset[j]['ifile'])

    s = dataset[i]['state'].numpy()  # state corresponds to the last image in the list!
    state_idx = np.where(s)[0]
    bx = retrieve_button(state_idx)

    world = dataset[i]['mario'][2].item()
    level = dataset[i]['mario'][3].item()

    if world != wrd:
        print("World -> " + str(world))
        wrd = world

    if level != lvl:
        print("======> Level -> " + str(level))
        lvl = level

    sample = {'noframes': noFrames, 'seq': i, 'world': world, 'level': level, 'image': ifile, 'state': dataset[i]['sfile'], 'bx': bx}
    bxAll.append(sample)
    # if (i + 1) % (print_every + 1) == 0:
    #     print(sample)

print("All: " + str(len(bxAll)))
save_csv(bx_data_path, "ALL", bxAll)
print("Done!")
