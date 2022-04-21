from operator import gt
from numpy import dtype
import radar.build.radar_processor
from os import listdir
from os.path import isfile, join
import numpy as np
import os


def get_groundtruth_odometry(gt_path, t1, t2):
    gt_data = []
    with open(gt_path, 'r') as f:
        f.readline()
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split(',')
            if (int(line[9]) == t1) and (int(line[8]) == t2):
                for j in range(2, 8):
                    gt_data.append(float(line[j]))
    return gt_data


root_path = '/home/gabs/Downloads/oxford_radar_robotcar_dataset_sample_medium/'
seqs = ['2019-01-10-14-36-48-radar-oxford-10k-partial']
# randomize ordering of image pairs

for seq in seqs:
    radar_path = root_path + seq + '/radar'
    gt_path = root_path + seq + '/gt/radar_odometry.csv'
    radar_files = [f for f in listdir(
        radar_path) if isfile(join(radar_path, f))]
    radar_files.sort(key=lambda f: int(f.split('.')[0]))
    correspondence_path = 'correspondences/' + seq
    try:
        os.mkdir(correspondence_path)
    except OSError as error:
        print(error)
    for i in range(1, len(radar_files) - 1):
        t1 = int(radar_files[i].split('.')[0])
        t2 = int(radar_files[i + 1].split('.')[0])
        gt_data = get_groundtruth_odometry(gt_path, t1, t2)
        if gt_data is not None:
            correspondeces = radar.build.radar_processor.get_radar_correspondeces(radar_path, radar_files[i - 1], radar_files[i], 1)
            correspondeces.append(np.array(gt_data))
        else:
            print("pair don't have GT")
        np.save(correspondence_path + '/' + 'pair_%d_%d.npy' %
                (t1, t2), correspondeces)
