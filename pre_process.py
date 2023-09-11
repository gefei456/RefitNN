#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         10:39
# @Author:      WGF
# @File:        pre_process.py
# @Description:

import time
import scipy.io as scio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from other_func import pos_screen2std, trial_cell_to_dict
import math
import scipy
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os
import pickle

DataTypeName = 'Matlab'
hold_time_bias = -0.0
stime_bias = 0.0
etime_bias = 0.0
SAVEFILE = False
screenWeight = 1280
screenHeight = 1024
spike_lag = -0.14
testIndex = 0
bin_size = 0.05
pre_bin_num = 3
isAlign = 'noAlign'

if __name__ == '__main__':

    if DataTypeName == 'Python':
        file_path = 'D:\Projects\Data\BJ_data_2023/20230629/'
        session_name = file_path.split('_')[-2] + '-' + file_path.split('_')[-1].split('/')[0]
        trialData = scio.loadmat(file_path + r"trialData.mat")  # 从文件中加载数据
        trials = trial_cell_to_dict(trialData)
        expData = scio.loadmat(file_path + r"expData.mat")  # 从文件中加载数据
        truthPoints = expData["truthPoints"]
        neuroData = scio.loadmat(file_path + r"neuroData.mat")
        manualPulses = neuroData["manualPulses"][0]  # 从文件中加载数据
        spike_data = neuroData["spike_data"][0]    #256*n
        posPulses = neuroData["posPulses"]

        truthPointsFixed = np.array([]).reshape(0, 3)
        # NOKOV 坐标和时间合并
        for i in range(min(posPulses.shape[1], truthPoints.shape[0])):
            truthPointsFixed = np.vstack((truthPointsFixed, np.hstack((posPulses[0, i], truthPoints[i, 1:3]))))

        # 有效trial
        valid_trials=[]
        for trial in trials:
            if trial["result"] == "success":
                valid_trials.append(trial)
        valid_trials = np.array(valid_trials)

        # trial数据预处理
        trial_fixed = []
        for trial in valid_trials:
            startTrial = trial["centerMouseDown"][0]
            endTrial = trial["targetMouseDown"][0]
            dirTrial = trial["ts_targetDisplay"][3]
            start_time = manualPulses[int(startTrial) * 2 - 1] + hold_time_bias + stime_bias
            end_time = manualPulses[int(endTrial) * 2 - 1] + etime_bias
            trial_fixed.append([start_time, end_time, dirTrial])
        trial_fixed = np.array(trial_fixed)
        print('valid trials:', len(trial_fixed))

    else:
        # file_path = 'D:\Projects\Data\SH_data_2023/20230209-3/'
        file_path = 'D:\Projects\Data\SH_data_2023/20230209-2/'
        dataPath = file_path + '20230209-2.mat'
        session_name = dataPath.split('.')[-2].split('/')[-1]
        truthPointsFixed = scio.loadmat(dataPath)['truthPointsFixed']
        spike_data = scio.loadmat(dataPath)['spike_data'][:, 0]
        trial_fixed = scio.loadmat(dataPath)['trialData']

        for i, trial in enumerate(trial_fixed):
            trial[0] += stime_bias
            trial[1] += etime_bias

# 数据分组
    successIndices = range(0, len(trial_fixed))
    poorIndices = []
    testSum = math.ceil(8)
    testIndices = range(testIndex, testIndex + testSum)
    trainIndices = [x for x in successIndices if x not in testIndices if x not in poorIndices]
    train_trials = trial_fixed[trainIndices]
    test_trials = trial_fixed[testIndices]

    if DataTypeName == 'Python':
        # 排除杂点
        valid_truth = []
        for pos in truthPointsFixed:
            if pos[1] != 6000:
                # 坐标转换
                pos_new = pos_screen2std(pos[1:3], screenWeight, screenHeight)
                pos[1:3] = pos_new
                valid_truth.append(pos[0:3])
        valid_truth = np.array(valid_truth)
    else:
        # 排除1125数据动捕杂点
        valid_truth = np.array(truthPointsFixed[~np.isnan(truthPointsFixed).any(axis=1)])

# TRAIN
    # Align
    if isAlign == 'Align':
        # tiral剪切对齐版本
        t_sum = 0
        for i, trial in enumerate(trial_fixed):
            t = trial[1] - trial[0]
            t_sum += t
        t_mean = int(t_sum / len(trial_fixed) / bin_size) * bin_size
        for i, trial in enumerate(trial_fixed):
            trial[0] = trial[1] - t_mean - bin_size / 2

    Time_new_list = np.array([]).reshape(0)
    Frame_new_list = np.array([]).reshape(0)
    HP_new_list = np.array([]).reshape(0, 2)
    X_Feature_list = []
    X_Truth_list = []
    for i, trial in enumerate(trial_fixed):
        X_Feature = []
        X_Truth = np.array([]).reshape(0, 4)
        start_time = trial[0]
        end_time = trial[1]

        bin_timeStamps = np.arange(start_time, end_time, bin_size)
        Time_new_list = np.hstack((Time_new_list, bin_timeStamps))

        # 从hand_pos查找timestamps对应点，插值计算
        HP_list = np.array([]).reshape(0, 2)
        for j in range(bin_timeStamps.size):
            cur_time = bin_timeStamps[j]
            time_list = valid_truth[:, 0]
            pos_x_list = valid_truth[:, 1]
            pos_y_list = valid_truth[:, 2]
            pos_x_inter = np.interp(cur_time, time_list, pos_x_list)
            pos_y_inter = np.interp(cur_time, time_list, pos_y_list)
            # HP_new_list = np.vstack((HP_new_list, [pos_x_inter, pos_y_inter]))
            HP_list = np.vstack((HP_list, [pos_x_inter, pos_y_inter]))

        HP_fix_list = np.array([]).reshape(0, 2)
        for j in range(HP_list.shape[0]):
            if np.sum(np.isnan(HP_list[j, :])) > 0:
                continue
            else:
                # pos_local = HP_list[j, :] - HP_list[0, :]
                pos_local = HP_list[j, :]
                HP_fix_list = np.vstack((HP_fix_list, pos_local))

        HP_len = len(HP_fix_list)
        vel_truth = np.zeros((HP_len - 1, 2))
        for j in range(HP_len):
            if j == HP_len - 1:
                break
            vel_truth[j, :] = (HP_fix_list[j + 1, :] - HP_fix_list[j, :]) \
                              / bin_size
        x_temp = np.hstack((HP_fix_list[0:-1], vel_truth))
        X_Truth = np.vstack((X_Truth, x_temp))
        X_Truth_list.append(X_Truth)
# 神经信号发放
        for j, t_stamp in enumerate(bin_timeStamps):
            if j == len(bin_timeStamps) - 1:
                break
            feat = []
            for k, data in enumerate(spike_data):
                zk_ch_list = []
                for m in range(pre_bin_num):
                    if data.shape != (0, 0):
                        zk_ch = np.sum((data >= (t_stamp + (m - pre_bin_num) * bin_size + spike_lag))
                                       & (data <= (t_stamp + (m - pre_bin_num + 1) * bin_size + spike_lag))) \
                                 / bin_size
                    else:
                        zk_ch = 0
                    zk_ch_list.append(zk_ch)
                zk_ch_list = np.array(zk_ch_list).reshape(-1, pre_bin_num)
                feat.append(zk_ch_list)
            feat = np.array(feat).reshape(-1, pre_bin_num)
            X_Feature.append(feat)
        X_Feature = np.array(X_Feature)
        X_Feature_list.append(X_Feature)

# 显示真实值轨迹
    for i, traj in enumerate(X_Truth_list):
        if i == 8:
            break
        plt.plot(traj[:, 0], traj[:, 1], 'r', linestyle='-', linewidth=0.7)
        vel_traj = np.cumsum(traj[:, 2:], axis=0) * bin_size
        plt.plot(vel_traj[:, 0], vel_traj[:, 1], 'b', linestyle='-', linewidth=0.7)
    plt.show()

# 保存
    file_to_save = {"truth_traj": X_Truth_list, "spike_list": X_Feature_list, "label_list": list(trial_fixed[:, 2])}
    with open(f"{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "wb") as f:
        pickle.dump(file_to_save, f)
    # np.save(f"{session_name}_ReNN.npy", file_to_save)
    print(f'成功保存{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl文件!')

# 读取
    with open(f"{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f:
        data_RefitNN = pickle.load(f)
    # print(data_RefitNN)


