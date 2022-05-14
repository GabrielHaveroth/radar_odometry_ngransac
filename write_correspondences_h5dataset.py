import json
import os
import h5py
import numpy as np

config_file_path = 'config/parameters.json'

with open(config_file_path, 'r') as f:
    configs = json.load(f)

total_data = sorted([subdir[0]
                     for subdir in os.walk(configs['data_path'])][1:])
train_folders = [total_data[seq] for seq in configs['train_seqs']]
val_folders = [total_data[seq] for seq in configs['val_seqs']]

# Listes to save files
files = []

DATASET_TYPE = 'val'
FILE_NAME = f'{DATASET_TYPE}.hdf5' 


if DATASET_TYPE == 'train':
    for folder in train_folders:
        files += [folder + '/' + f for f in os.listdir(folder)]
elif DATASET_TYPE == 'val':
    for folder in val_folders:
        files += [folder + '/' + f for f in os.listdir(folder)]

f = h5py.File(FILE_NAME, 'w')

for ix, file in enumerate(files):
    data = np.load(file, allow_pickle=True)
    grp = f.create_group(f'pair{ix}')
    pts1 = data[0]
    kp1 = data[1]
    pts2 = data[2]
    kp2 = data[3]
    ratios = data[4]
    gt = data[5]
    dataset = dict(pts1=pts1, kp1=kp1, pts2=pts2,
                   kp2=kp2, ratios=ratios, gt=gt)
    for elem in dataset:
        grp.create_dataset(elem, data=dataset[elem])
f.close()