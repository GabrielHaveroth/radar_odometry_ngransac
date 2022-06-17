from util import get_transform
import torch
import argparse
import json
import os
from network import CNNet
import ngransac
from os.path import isfile, join
import numpy as np

torch.cuda.empty_cache()

arg_parser = argparse.ArgumentParser(description='json config file path')

# Add the arguments
arg_parser.add_argument('config',
                        metavar='config_path',
                        type=str,
                        help='the path to config file')


config_file_path = arg_parser.parse_args().config

with open(config_file_path, 'r') as f:
    configs = json.load(f)

with open(config_file_path, 'r') as f:
    total_data = sorted([subdir[0]
                         for subdir in os.walk(configs['data_path'])][1:])
    configs = json.load(f)
seq = 1
seqs = [total_data[seq] for seq in configs['train_seqs']]
files = []

for seq in seqs:
    files = [f for f in os.listdir(seq) if isfile(join(seq, f))]
    files.sort(key=lambda f: int(f.split('_')[1]))
    print(len(files))
    for file in files:
        data = np.load(seq + '/' + file, allow_pickle=True)
        print(data[0].shape)
        break