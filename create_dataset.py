#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         16:32
# @Author:      WGF
# @File:        create_dataset.py
# @Description:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, df_feature, df_pos, df_vel):

        assert len(df_feature) == len(df_pos)
        assert len(df_feature) == len(df_vel)

        # df_feature = df_feature.reshape(df_feature.shape[0], df_feature.shape[1] // 6, df_feature.shape[2] * 6)
        self.df_feature = df_feature
        self.df_vel = df_vel
        self.df_pos = df_pos

        self.df_feature = torch.tensor(
            self.df_feature, dtype=torch.float32)
        self.df_vel = torch.tensor(
            self.df_vel, dtype=torch.float32)
        self.df_pos = torch.tensor(
            self.df_pos, dtype=torch.float32)

    def __getitem__(self, index):
        sample, pos, vel = self.df_feature[index], self.df_pos[index], self.df_vel[index]
        return sample, pos, vel

    def __len__(self):
        return len(self.df_feature)


def createDataset(dict_file):
    # file_to_save = {"truth_traj": X_Truth, "spike_list": X_Feature}
    assert len(dict_file['spike_list']) == len(dict_file['truth_traj'])
    lst = list(range(len(dict_file['spike_list'])))  # 创建长度为1298的列表
    ratio = [0.8, 0.2]  # 切分比例，第一部分占80%，第二部分占20%
    split_index = int(len(lst) * ratio[0])  # 计算切分索引
    train_feat_list = dict_file['spike_list'][:split_index]
    test_feat_list = dict_file['spike_list'][split_index:]
    train_traj_list = dict_file['truth_traj'][:split_index]
    test_traj_list = dict_file['truth_traj'][split_index:]
    train_feat = torch.tensor(np.concatenate(train_feat_list, axis=0)).to(device)
    test_feat = torch.tensor(np.concatenate(test_feat_list, axis=0)).to(device)
    train_traj = torch.tensor(np.concatenate(train_traj_list, axis=0)).to(device)
    test_traj = torch.tensor(np.concatenate(test_traj_list, axis=0)).to(device)
    train_dataset = CustomDataset(train_feat, train_traj[:, :2], train_traj[:, 2:])
    test_dataset = CustomDataset(test_feat, test_traj[:, :2], test_traj[:, 2:])
    return train_dataset, test_dataset


if __name__ == '__main__':
    session_name = '20230209_3'

    with open(f"{session_name}_ReNN.pkl", "rb") as f:
        data_RefitNN = pickle.load(f)

    train_dataset, test_dataset = createDataset(data_RefitNN)
    print('train_dataset:', train_dataset.df_feature.shape)
    print('test_dataset:', test_dataset.df_feature.shape)
