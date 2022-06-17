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
            else:
                gt_data = None
    return gt_data


def save_correspondences_pair_gt(pair):
    t1, t2 = pair['times']
    gt_path = pair['gt_path']
    correspondence_path = pair['correspondence_path']
    gt_data = get_groundtruth_odometry(gt_path, t1, t2)

    if gt_data is not None:
        if len(gt_data) != 6:
            correspondences = radar.build.radar_processor.get_radar_correspondences(pair['radar_path'], pair['radar_imgs'][0], pair['radar_imgs'][1], 1)
            correspondences.append(np.array(gt_data))
            np.save(correspondence_path + '/' + 'pair_%d_%d.npy' % (t1, t2), correspondences)
    else:
        print("pair don't have GT")


def create_pairs(seq):
    pairs_data = []
    radar_path = root_path + seq + '/radar'
    gt_path = root_path + seq + '/gt/radar_odometry.csv'
    correspondence_path = '/home/gabs/datasets/correspondences_cen2018/' + seq
    try:
        os.mkdir(correspondence_path)
    except OSError as error:
        print(error)
    radar_files = [f for f in listdir(radar_path) if isfile(join(radar_path, f))]
    radar_files.sort(key=lambda f: int(f.split('.')[0]))

    for i in range(0, len(radar_files) - 1):
        if i > 0:
            t1 = int(radar_files[i].split('.')[0])
            t2 = int(radar_files[i + 1].split('.')[0])
            pairs_data.append({'radar_imgs': (radar_files[i - 1], radar_files[i]),
                               'times': (t1, t2),
                               'radar_path': radar_path,
                               'gt_path': gt_path,
                               'correspondence_path': correspondence_path})
  
    return pairs_data


root_path = '/mnt/ubuntu_2004/home/gabs/datasets/'
# seqs = ['2019-01-10-11-46-21-radar-oxford-10k',
#         '2019-01-10-12-32-52-radar-oxford-10k',
#         '2019-01-10-14-02-34-radar-oxford-10k',
#         '2019-01-10-14-36-48-radar-oxford-10k-partial',
#         '2019-01-10-14-50-05-radar-oxford-10k',
#         '2019-01-10-15-19-41-radar-oxford-10k',
#         '2019-01-11-12-26-55-radar-oxford-10k',
#         '2019-01-11-13-24-51-radar-oxford-10k',
#         '2019-01-11-14-02-26-radar-oxford-10k',
#         '2019-01-11-14-37-14-radar-oxford-10k',
#         '2019-01-14-12-05-52-radar-oxford-10k',
#         '2019-01-14-12-41-28-radar-oxford-10k',
#         '2019-01-14-13-38-21-radar-oxford-10k',
#         '2019-01-14-14-15-12-radar-oxford-10k',
#         '2019-01-14-14-48-55-radar-oxford-10k',
#         '2019-01-15-12-01-32-radar-oxford-10k',
#         '2019-01-15-12-52-32-radar-oxford-10k-partial',
#         '2019-01-15-13-06-37-radar-oxford-10k',
#         '2019-01-15-13-53-14-radar-oxford-10k',
#         '2019-01-15-14-24-38-radar-oxford-10k',
#         '2019-01-16-11-53-11-radar-oxford-10k',
#         '2019-01-16-13-09-37-radar-oxford-10k',
#         '2019-01-16-13-42-28-radar-oxford-10k',
#         '2019-01-16-14-15-33-radar-oxford-10k',
#         '2019-01-17-11-46-31-radar-oxford-10k',
#         '2019-01-17-12-48-25-radar-oxford-10k',
#         '2019-01-17-13-26-39-radar-oxford-10k',
#         '2019-01-17-14-03-00-radar-oxford-10k',
#         '2019-01-18-12-42-34-radar-oxford-10k',
#         '2019-01-18-14-14-42-radar-oxford-10k',
#         '2019-01-18-14-46-59-radar-oxford-10k',
#         '2019-01-18-15-20-12-radar-oxford-10k']

seqs = ['2019-01-10-14-36-48-radar-oxford-10k-partial']
for seq in seqs:
    pairs = create_pairs(seq)
    print(len(pairs))
    for pair in pairs:
        save_correspondences_pair_gt(pair)
