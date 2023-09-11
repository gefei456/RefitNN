#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         16:12
# @Author:      WGF
# @File:        trainRefitNN.py
# @Description:
import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from RefitNN import FC4L256Np05_CNN1L16N_SBP
from create_dataset import createDataset
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
# from torchsummary import summary
from pytorch_model_summary import summary
import sklearn.decomposition as skd
from torch.utils.tensorboard import SummaryWriter
import shutil
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


def train_NN(dataloader, model, loss_fn, optimizer, label_state):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    num_batches = 0
    for i, (X, pos, vel) in enumerate(dataloader):
        # Compute prediction and loss
        # y_norm = nn.BatchNorm1d(y)
        pred = model(X)
        if label_state == 'vel':
            loss = loss_fn(pred, vel)
        elif label_state == 'pos':
            loss = loss_fn(pred, pos)
        else:
            loss = loss_fn(pred, pos + vel)
        total_loss += loss.item()
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        num_batches += 1
        if i % 10 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss = total_loss / num_batches
    print(f"Train loss:\n {train_loss:>8f} \n")
    return train_loss


def test_NN(dataloader, model, loss_fn, label_state):
    model.eval()
    # size = len(dataloader.dataset)
    num_batches = 0
    # batch_size = dataloader.batch_size
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for i, (X, pos, vel) in enumerate(dataloader):
            pred = model(X)
            if label_state == 'vel':
                loss = loss_fn(pred, vel)
            elif label_state == 'pos':
                loss = loss_fn(pred, pos)
            else:
                loss = loss_fn(pred, pos + vel)
            test_loss += loss.item()
            num_batches += 1
    test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test loss:\n {test_loss:>8f} \n")
    return test_loss


if __name__ == '__main__':
    totalStartTime = time.time_ns()
    learning_rate = 1e-5
    input_size = 256
    PCA_ncomp = input_size
    hidden_size = [256, 256, 256]
    ConvSizeOut = 16  # 16 -> 3
    ConvSize = 3  # 3 -> 1
    label_state = 'vel'
    if label_state == 'vel' or 'pos':
        num_states = 2
    else:
        num_states = 4
    batchSize = 1024
    session_name = '20230209-3'
    bin_size = 0.05
    only4Test = True
    torch.manual_seed(13)
    spike_lag = -0.14
    isAlign = 'noAlign'
    isMerged = True
    isPCA = False
    epochs = 10000
    patience = 1000

    # load pkl file
    if isMerged:
        session_name1 = '20230209-2'
        session_name2 = '20230209-3'
        with open(f"{session_name1}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f1:
            data_RefitNN1 = pickle.load(f1)
        with open(f"{session_name2}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f2:
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
        with open(f"{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f:
            data_RefitNN = pickle.load(f)

    # 创建多输出回归模型，使用SelectKBest作为特征选择器，同时处理两个相关目标
    model = MultiOutputRegressor(estimator=SelectKBest(score_func=f_regression, k=32))
    # model = MultiOutputRegressor(estimator=RandomForestRegressor(n_estimators=100, random_state=42))
    # 训练模型
    X = data_spike_merge[:, :, :, 2:].reshape(-1, 256)
    y = data_traj_merge[:,:,2:].reshape(-1,2)
    model.fit(X, y)
    # 获取特征选择得分
    feature_scores = model.estimators_[0].scores_ + model.estimators_[1].scores_  # 平均综合两个目标的得分
    # feature_scores = model.estimators_[0].feature_importances_ + model.estimators_[1].feature_importances_
    # 对得分进行排序并获取排序后的索引
    np.nan_to_num(feature_scores, copy=False, nan=-np.inf)
    sorted_indices = np.argsort(feature_scores)[::-1]

    # 选择前十个特征的索引
    top_n_indices = sorted_indices[:32]

    # 简化X，仅保留前十个特征
    X_simplified = X[:, top_n_indices]

    # 打印特征选择得分和对应的特征序号
    print("特征选择得分 (综合考虑两个相关目标)：")
    for i, idx in enumerate(sorted_indices):
        print(f"{idx}: {feature_scores[idx]}")
    print("选择后")
    for i, idx in enumerate(top_n_indices):
        print(f"{idx}: {feature_scores[idx]}")

    # # 输出互信息结果
    # for i, mi in enumerate(mutual_info):
    #     print(f"特征 {i + 1}: {mi:.2f}")

    local_time = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
    print(local_time)
    exp_path = f'./{session_name}_EXP/{local_time}/'
    writer = SummaryWriter(exp_path + 'loss')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isPCA:
        ##############################  PCA  #########################
        # stime4test = time.time_ns()
        pca = skd.PCA(n_components=PCA_ncomp)
        # data_spike_merge = np.array(data_spike_merge)
        data_spike_merge = data_spike_merge[:,:,:,2:]
        data_spike_merge = np.reshape(data_spike_merge,
                                      (np.prod(data_spike_merge.shape[0:2]), np.prod(data_spike_merge.shape[2::])))
        pca.fit(data_spike_merge)
        print('total explained variance ratio: ', np.cumsum(pca.explained_variance_ratio_)[-1])
        data_spike_merge_extr = pca.transform(data_spike_merge)
        data_spike_merge_extr = np.reshape(data_spike_merge_extr, (len(data_label_merge), -1, PCA_ncomp))
        data_RefitNN['spike_list'] = list(data_spike_merge_extr)
        # print("运算所需时间", (time.time_ns() - stime4test) / 1e9)
    ######################### Align Batch #######################
    spike_data_list = data_RefitNN['spike_list']
    bin_len = spike_data_list[0].shape[0]
    print('bin_len:', bin_len)
    # batchSize = 1 * bin_len
    #############################################################
    if not only4Test:
    # train model
        # Define the network
        ReftiNN_model = FC4L256Np05_CNN1L16N_SBP(input_size, hidden_size, ConvSize, ConvSizeOut, num_states).to(device)
        print('Model:', ReftiNN_model)

        # loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.L1Loss()
        # loss_fn = nn.SmoothL1Loss(beta=0.8)
        loss_fn = nn.MSELoss()
        # loss_fn = torch.nn.HuberLoss(reduction='mean', delta=25)
        # optimizer = torch.optim.SGD(ReftiNN_model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(ReftiNN_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

        train_dataset, test_dataset = createDataset(data_RefitNN)

        train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, drop_last=True)
        best_val_loss = float('inf')
        num_bad_epochs = 0
        train_loss_list = []
        val_loss_list = []
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loss = train_NN(train_dataloader, ReftiNN_model, loss_fn, optimizer, label_state)
            val_loss = test_NN(test_dataloader, ReftiNN_model, loss_fn, label_state)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            writer.add_scalars('Loss', {'trainLoss': train_loss, 'valLoss': val_loss}, t)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
            if num_bad_epochs >= patience:
                break
        print("Done!")
        print(f"训练所需总时间{(time.time_ns() - totalStartTime) / 1e9}s")
        plt.figure(3)
        plt.plot(range(len(train_loss_list)), train_loss_list, 'r', '--')
        plt.plot(range(len(val_loss_list)), val_loss_list, 'b', '--')


        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        # save loss
        writer.close()
        # save model
        model_path = exp_path + 'RefitNN-Model.pth'
        state_dict_path = exp_path + 'RefitNN-state_dict.pt'
        # torch.save(ReftiNN_model.state_dict(), model_path)
        torch.save(ReftiNN_model, model_path)
        torch.save({
            'learning_rate': learning_rate,
            'inputSize': input_size,
            'hiddenSize': hidden_size,
            'batchSize': batchSize,
            'ConvSizeOut': ConvSizeOut,
            'ConvSize': ConvSize,
            'bin_size': bin_size,
            'label_state': label_state,
            # 'optimizer_param_groups': optimizer.param_groups,
            'loss_type': type(loss_fn).__name__,
            'optimizer_type': type(optimizer).__name__
        }, state_dict_path)
    else:
        model_path = 'good_RefitNN-2023-08-25--11_16_41.pth'
        state_dict_path = 'good_RefitNN-2023-08-25--11_16_41-state_dict.pt'
        shutil.rmtree(exp_path)
# load model
#     ReftiNN_model = FC4L256Np05_CNN1L16N_SBP(input_size, hidden_size, ConvSize, ConvSizeOut, num_states).to(device)
    ReftiNN_model = torch.load(model_path).to(device)
    checkpoints = torch.load(state_dict_path)
    print('checkpoints:', checkpoints)
    # ReftiNN_model.load_state_dict(torch.load(model_path))
    ReftiNN_model.eval()
    ReftiNN_model.requires_grad_(False)
    # print(ReftiNN_model.__module__)
    print(ReftiNN_model)
    if isPCA:
        print(summary(ReftiNN_model, torch.zeros(batchSize, PCA_ncomp).to(device), show_input=True,
                      show_hierarchical=True))
    else:
        print(summary(ReftiNN_model, torch.zeros(batchSize, PCA_ncomp, ConvSize).to(device), show_input=True,
                      show_hierarchical=True))
    # summary(ReftiNN_model, (64, 1))

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
        # pred_vel = ReftiNN_model(X).to('cpu')
        # test real time predict
        pred_vel = torch.tensor([])
        for j, xx in enumerate(X):
            # stime4test = time.time_ns()
            pred_vel = torch.cat([pred_vel, ReftiNN_model(torch.unsqueeze(xx, dim=0)).to('cpu')])
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
