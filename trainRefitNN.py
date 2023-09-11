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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def train_NN(dataloader, model, loss_fn, optimizer, label_state):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    num_batches = 0
    for i, (X, pos, vel) in enumerate(dataloader):
        # Compute prediction and loss
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
    num_batches = 0
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
    print(f"Test loss:\n {test_loss:>8f} \n")
    return test_loss


if __name__ == '__main__':
    isDebugMode = False
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
    path_head = './data_backup/'

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
    exp_path = f'./{session_name}_EXP/{local_time}/'
    writer = SummaryWriter(exp_path + 'loss')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isDebugMode:
        print(local_time)
        spike_data_list = data_RefitNN['spike_list']
        bin_len = spike_data_list[0].shape[0]
        print('bin_len:', bin_len)

    # train model
    ReftiNN_model = FC4L256Np05_CNN1L16N_SBP(input_size, hidden_size, ConvSize, ConvSizeOut, num_states).to(device)
    if isDebugMode:
        print('Model:', ReftiNN_model)
    loss_fn = nn.MSELoss()
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

    # plot
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
