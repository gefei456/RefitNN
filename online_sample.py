#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         10:55
# @Author:      WGF
# @File:        online_sample.py
# @Description:

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from RefitNN import FC4L256Np05_CNN1L16N_SBP
from create_dataset import createDataset
import pickle
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
import matplotlib

if __name__ == '__main__':
    input_size = 256
    PCA_ncomp = input_size
    ConvSize = 3  # 3 -> 1
    label_state = 'vel'
    batchSize = 1024
    session_name = '20230209-3'
    bin_size = 0.05
    spike_lag = -0.14
    isAlign = 'noAlign'
    path_head = 'data_backup/'
    isMerged = True

    # load pkl file
    if isMerged:
        session_name1 = '20230209-2'
        session_name2 = '20230209-3'
        with open(path_head + f'{session_name1}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl', 'rb') as f1:
            data_RefitNN1 = pickle.load(f1)
        with open(path_head + f'{session_name2}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl', 'rb') as f2:
            data_RefitNN2 = pickle.load(f2)

        data_spike_merge = data_RefitNN1['spike_list'] + data_RefitNN2['spike_list']
        data_label_merge = data_RefitNN1['label_list'] + data_RefitNN2['label_list']
        data_traj_merge = data_RefitNN1['truth_traj'] + data_RefitNN2['truth_traj']
        data_RefitNN = {'truth_traj': data_traj_merge, 'spike_list': data_spike_merge, 'label_list': data_label_merge}
        trial_length = np.array([x.shape[0] for x in data_spike_merge])
        trial_max_length = np.max(trial_length)
        data_spike_merge = np.array(
            [np.pad(x, ((trial_max_length - x.shape[0], 0), (0, 0), (0, 0)), 'constant') for x in data_spike_merge])
        data_traj_merge = np.array(
            [np.pad(x, ((trial_max_length - x.shape[0], 0), (0, 0)), 'constant') for x in data_traj_merge])
        data_RefitNN = {'truth_traj': list(data_traj_merge), 'spike_list': data_spike_merge,
                        'label_list': data_label_merge}
        session_name = session_name1 + '_' + session_name2
    else:
        with open(path_head + f'{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl', 'rb') as f:
            data_RefitNN = pickle.load(f)
    local_time = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = '20230209-2_20230209-3_EXP/good_2023-08-25/RefitNN-Model.pth'
    state_dict_path = '20230209-2_20230209-3_EXP/good_2023-08-25/RefitNN-state_dict.pt'
# load model
    ReftiNN_model = torch.load(model_path).to(device)
    # checkpoints = torch.load(state_dict_path)
    # print('checkpoints:', checkpoints)
    ReftiNN_model.eval()
    ReftiNN_model.requires_grad_(False)
    # print(ReftiNN_model)

    print(summary(ReftiNN_model, torch.zeros(batchSize, PCA_ncomp, ConvSize).to(device), show_input=True,
                      show_hierarchical=True))
    trial_num = len(data_RefitNN['truth_traj'])
    trial_sec_start = trial_num - 40
    trial_sec_end = trial_num - 0
    test_traj_list = data_RefitNN['truth_traj'][trial_sec_start:trial_sec_end]
    test_feat_list = data_RefitNN['spike_list'][trial_sec_start:trial_sec_end]
    test_label_list = data_RefitNN['label_list'][trial_sec_start:trial_sec_end]

    # plot
    plt.figure(2)
    colorlist = list(matplotlib.colors.TABLEAU_COLORS)
    linetypelist = ['-', '--', '-.', ':']
    markerlist = ['.', ',', 'o', '2', '1']
    for i, feat in enumerate(test_feat_list):
        X = torch.tensor(feat, dtype=torch.float32).to(device)
        # test real time predict
        pred_vel = torch.tensor([])
        for j, xx in enumerate(X):
            # stime4test = time.time_ns()
            vel_val = torch.unsqueeze(xx, dim=0)
            predict_vel = ReftiNN_model(vel_val).to('cpu')
            pred_vel = torch.cat([pred_vel, predict_vel])
            # print("运算所需时间", (time.time_ns() - stime4test) / 1e9)
        if label_state == 'vel':
            true_traj = np.cumsum(test_traj_list[i][:, 2:], axis=0) * bin_size
            pred_traj = np.cumsum(pred_vel.numpy(), axis=0) * bin_size
        elif label_state == 'pos':
            true_traj = test_traj_list[i][:, :2]
            pred_traj = pred_vel.numpy()
        else:
            # true_traj = np.cumsum(test_traj_list[i][:, 2:], axis=0) * bin_size
            # pred_traj = np.cumsum(pred_vel.numpy(), axis=0) * bin_size
            true_traj = test_traj_list[i][:, :2]
            pred_traj = pred_vel.numpy()
        plt.plot(true_traj[:, 0], true_traj[:, 1], colorlist[int(test_label_list[i]) - 1], linestyle='-', linewidth=0.7)
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], colorlist[int(test_label_list[i]) - 1], linestyle='--', linewidth=0.7)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.title(f'RefitNN-{session_name}-{local_time}')
    plt.show()
