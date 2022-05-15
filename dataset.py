import numpy as np
import torch
import os
import math
import util
from torch.utils.data import Dataset
import h5py


class RadarCorrespondences(Dataset):
    """Sparse correspondences radar from radar Oxford dataset."""

    def __init__(self, folders, ratiothreshold, nfeatures,
                 overwrite_side_info=False):

        # ensure fixed number of features, -1 keeps original feature count
        self.nfeatures = nfeatures
        self.ratiothreshold = ratiothreshold  # threshold for Lowe's ratio filter
        # if true, provide no side information to the neural guidance network
        self.overwrite_side_info = overwrite_side_info

        # collect precalculated correspondences of all provided datasets
        self.files = []
        for folder in folders:
            self.files += [folder + '/' + f for f in os.listdir(folder)]
        self.minset = 2  # minimal set for SVD

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # load precalculated correspondences
        data = np.load(self.files[idx], allow_pickle=True)

        # correspondence coordinates and matching ratios (side information)
        pts1 = torch.from_numpy(data[0])
        kp1 = np.array(np.float32([data[1]]))
        pts2 = torch.from_numpy(data[2])
        kp2 = np.array(np.float32([data[3]]))
        ratios = np.array([data[4]])
        gt = data[5]
        # image sizes
        im_size1 = im_size2 = np.array((964, 964))
        # ground truth pose
        # applying Lowes ratio criterion
        ratio_filter = ratios[0, :, 0] < self.ratiothreshold
        if ratio_filter.sum() < self.minset:  # ensure a minimum count of correspondences
            print("WARNING! Ratio filter too strict. Only %d correspondences would be left, so I skip it." % int(
                ratio_filter.sum()))
        else:
            kp1 = kp1[:, ratio_filter, :]
            kp2 = kp2[:, ratio_filter, :]
            ratios = ratios[:, ratio_filter, :]

        if self.overwrite_side_info:
            ratios = np.zeros(ratios.shape, dtype=np.float32)

        # For fundamental matrices, normalize image coordinates using the
        # image size (network should be independent to resolution)
        util.normalize_pts(kp1, im_size1)
        util.normalize_pts(kp2, im_size2)
        # Stack image coordinates and side information into one tensor
        correspondences = np.concatenate((kp1, kp2, ratios), axis=2)
        correspondences = np.float32(np.transpose(correspondences))
        correspondences = torch.from_numpy(correspondences)

        if self.nfeatures > 0:
            # Ensure that there are exactly nfeatures entries in the data
            # tensor
            if correspondences.size(1) > self.nfeatures:
                rnd = torch.randperm(correspondences.size(1))
                correspondences = correspondences[:, rnd, :]
                correspondences = correspondences[:, 0:self.nfeatures]
                pts1 = pts1[:, rnd]
                pts1 = pts1[:, 0:self.nfeatures]
                pts2 = pts2[:, rnd]
                pts2 = pts2[:, 0:self.nfeatures]

            if correspondences.size(1) < self.nfeatures:
                result = correspondences
                pts1_fixed = pts1
                pts2_fixed = pts2
                for i in range(0, math.ceil(self.nfeatures / correspondences.size(1) - 1)):
                    rnd = torch.randperm(correspondences.size(1))
                    result = torch.cat(
                        (result, correspondences[:, rnd, :]), dim=1)
                    pts1_fixed = torch.cat((pts1_fixed, pts1[:, rnd]), dim=1)
                    pts2_fixed = torch.cat((pts2_fixed, pts2[:, rnd]), dim=1)

                correspondences = result[:, 0:self.nfeatures]
                pts1 = pts1_fixed[:, 0:self.nfeatures]
                pts2 = pts2_fixed[:, 0:self.nfeatures]
        T12 = np.float32(util.get_transform(gt[0], gt[1], gt[5]))
        return correspondences, T12, pts1, pts2


class RadarCorrespondencesHdf5(Dataset):
    """Sparse correspondences radar from radar Oxford dataset."""

    def __init__(self, hdf5_file, ratiothreshold, nfeatures,
                 overwrite_side_info=False):

        # ensure fixed number of features, -1 keeps original feature count
        self.nfeatures = nfeatures
        self.ratiothreshold = ratiothreshold  # threshold for Lowe's ratio filter
        # if true, provide no side information to the neural guidance network
        self.overwrite_side_info = overwrite_side_info

        # collect precalculated correspondences of all provided datasets
        self.files = []
        self.hdf5_data = h5py.File(hdf5_file, 'r')
        self.pairs = [group for group in self.hdf5_data.keys()]
        self.minset = 2  # minimal set for SVD

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        # load precalculated correspondences
        grp = self.hdf5_data[self.pairs[idx]]
        
        # correspondence coordinates and matching ratios (side information)
        pts1 = torch.from_numpy(grp['pts1'][:])
        kp1 = np.array(np.float32([grp['kp1'][:]]))
        pts2 = torch.from_numpy(grp['pts2'][:])
        kp2 = np.array(np.float32([grp['kp2'][:]]))
        ratios = np.array([grp['ratios'][:]])
        gt = grp['gt'][:]
        # image sizes
        im_size1 = im_size2 = np.array((964, 964))
        # ground truth pose
        # applying Lowes ratio criterion
        ratio_filter = ratios[0, :, 0] < self.ratiothreshold
        if ratio_filter.sum() < self.minset:  # ensure a minimum count of correspondences
            print("WARNING! Ratio filter too strict. Only %d correspondences would be left, so I skip it." % int(
                ratio_filter.sum()))
        else:
            kp1 = kp1[:, ratio_filter, :]
            kp2 = kp2[:, ratio_filter, :]
            ratios = ratios[:, ratio_filter, :]

        if self.overwrite_side_info:
            ratios = np.zeros(ratios.shape, dtype=np.float32)

        # For fundamental matrices, normalize image coordinates using the
        # image size (network should be independent to resolution)
        util.normalize_pts(kp1, im_size1)
        util.normalize_pts(kp2, im_size2)
        # Stack image coordinates and side information into one tensor
        correspondences = np.concatenate((kp1, kp2, ratios), axis=2)
        correspondences = np.float32(np.transpose(correspondences))
        correspondences = torch.from_numpy(correspondences)

        if self.nfeatures > 0:
            # Ensure that there are exactly nfeatures entries in the data
            # tensor
            if correspondences.size(1) > self.nfeatures:
                rnd = torch.randperm(correspondences.size(1))
                correspondences = correspondences[:, rnd, :]
                correspondences = correspondences[:, 0:self.nfeatures]
                pts1 = pts1[:, rnd]
                pts1 = pts1[:, 0:self.nfeatures]
                pts2 = pts2[:, rnd]
                pts2 = pts2[:, 0:self.nfeatures]

            if correspondences.size(1) < self.nfeatures:
                result = correspondences
                pts1_fixed = pts1
                pts2_fixed = pts2
                for i in range(0, math.ceil(self.nfeatures / correspondences.size(1) - 1)):
                    rnd = torch.randperm(correspondences.size(1))
                    result = torch.cat(
                        (result, correspondences[:, rnd, :]), dim=1)
                    pts1_fixed = torch.cat((pts1_fixed, pts1[:, rnd]), dim=1)
                    pts2_fixed = torch.cat((pts2_fixed, pts2[:, rnd]), dim=1)

                correspondences = result[:, 0:self.nfeatures]
                pts1 = pts1_fixed[:, 0:self.nfeatures]
                pts2 = pts2_fixed[:, 0:self.nfeatures]
        T12 = np.float32(util.get_transform(gt[0], gt[1], gt[5]))
        return correspondences, T12, pts1, pts2
