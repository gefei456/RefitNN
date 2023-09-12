#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         15:59
# @Author:      WGF
# @File:        RefitNN.py
# @Description:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Helper Function
def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)


# Definition of the neural network
class FC4L256Np05_CNN1L16N_SBP(nn.Module):
    def __init__(self, input_size=256, hidden_size=[256, 128, 16], ConvSize=3, ConvSizeOut=16, num_states=2):
        super().__init__()
        # assign layer objects to class attributes
        self.bn0 = nn.BatchNorm1d(input_size)
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_size * ConvSizeOut)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size[0])
        self.do1 = nn.Dropout(p=0.5)
        self.bn2 = nn.BatchNorm1d(hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.do2 = nn.Dropout(p=0.3)
        self.bn3 = nn.BatchNorm1d(hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.do3 = nn.Dropout(p=0.2)
        self.bn4 = nn.BatchNorm1d(hidden_size[2])
        self.fc4 = nn.Linear(int(hidden_size[2]), num_states)
        # self.fc4 = nn.Linear(hidden_size[1], hidden_size[2])
        # self.do4 = nn.Dropout(p=0.2)
        # self.bn5 = nn.BatchNorm1d(num_states)

        # self.fc6 = nn.Linear(int(hidden_size[2]), num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc6.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)
        # nn.init.zeros_(self.fc6.bias)
        # nn.init.zeros_(self.fc5.bias)

    def forward(self, x, BadChannels=[]):
        # forward always defines connectivity
        # x[:, BadChannels, :] = 0
        # x = self.bn0(x)
        # x = self.cn1(x.permute(0, 2, 1))
        # x = torch.tanh(self.bn1(flatten(x)))
        # x = torch.tanh(self.bn2(self.do1(self.fc1(x))))
        # x = torch.tanh(self.bn3(self.do2(self.fc2(x))))
        # x = torch.tanh(self.bn4(self.do3(self.fc3(x))))
        # scores = self.fc4(x)
        # # scores = (self.bn5(self.fc4(x)) - self.bn5.bias)/self.bn5.weight
        # # scores = self.bn5(self.fc4(x)) * self.bn5.weight + self.bn5.bias
        # # scores = self.bn5(self.fc4(x)) * self.bn5.running_mean + self.bn5.running_var

        x[:, BadChannels, :] = 0
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(flatten(x))
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = F.relu(self.do3(self.fc3(x)))
        # x = F.relu(self.do4(self.fc4(x)))
        # scores = self.fc6(x)
        scores = self.fc4(x)

        # # PCA 256 * 3 -> 64
        # x = self.bn0(x)
        # x = self.cn1(torch.unsqueeze(x, dim=1))
        # x = F.relu(self.bn1(flatten(x)))
        # x = F.relu(self.bn2(self.do1(self.fc1(x))))
        # x = F.relu(self.bn3(self.do2(self.fc2(x))))
        # x = F.relu(self.bn4(self.do3(self.fc3(x))))
        # scores = (self.bn5(self.fc4(x)) - self.bn5.bias)/self.bn5.weight

        return scores

    def predict(self, X, weight=0, target_pos=None):
        '''
        模型预测接口，包含refit模式（TODO..）
        :param X:
        :param weight:
        :param target_pos:
        :return:
        '''
        # if target_pos != None:
        #     X_new = refitVelocity(target_pos, X, weight)
        #     self.train()
        #     self.requires_grad_(True)
        #     return self.forward(X_new).to('cpu')
        # else:
        return self.forward(X)


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

def initModel(model_path:str, state_dict_path:str, isDebug=False, test=True):
    '''
    模型初始化接口
    :param model_path: 待加载模型路径
    :param state_dict_path: 待加载模型训练参数
    :param isDebug: debug模式
    :param test: 测试（。。）
    :return:
    '''
    ReftiNN_model = torch.load(model_path).to(device)
    checkpoints = torch.load(state_dict_path)
    if isDebug:
        print('checkpoints:', checkpoints)
    ReftiNN_model.eval()
    ReftiNN_model.requires_grad_(False)
    if test:
        vel_val = torch.zeros((1, 256, 3)).to(device)
        preV = ReftiNN_model(vel_val).to('cpu')
    return ReftiNN_model

import math as mt
def refitVelocity(target_pos, X, weight):
    '''
    refit计算旋转角度
    :param target_pos: 目标位置坐标
    :param x_state_true: 输入状态
    :param weight: 旋转权重
    :return: 更新后速度向量
    '''
    # 预测的位置和速度
    p_loc = X[0:2].reshape(-1)
    p_vel = X[2:4].reshape(-1)
    speed = np.sqrt(sum((p_vel ** 2)))
    correctVect = target_pos - p_loc
    # 计算target方向的tvx tvy
    vRot = speed * correctVect / np.linalg.norm(correctVect)
    # dot 点乘求和, target方向与预测速度的角度
    angle = np.arccos((vRot @ p_vel) / (np.linalg.norm(vRot) * np.linalg.norm(p_vel)))
    # 根据叉乘判断target方向与预测速度的旋转方向
    # 设矢量P = (x1, y1) ，Q = (x2, y2), P × Q = x1 * y2 - x2 * y1 。若P × Q > 0, 则P
    # 在Q的顺时针方向.若 P × Q < 0, 则P在Q的逆时针方向；若P × Q = 0, 则P与Q共线，但可能同向也可能反向；
    pos = p_vel[0] * vRot[1] - p_vel[1] * vRot[0]
    if pos < 0:
        angle = -angle
    # 根据权重计算需要旋转的角度
    angle = angle * weight
    # 坐标转换矩阵
    TranM = [[mt.cos(angle), mt.sin(angle)],
             [-mt.sin(angle), mt.cos(angle)]]
    # 向量旋转angle角度
    new_vel = p_vel @ TranM
    # new_vel = TranM @ p_vel
    x_new = np.vstack((X[0:2, :], new_vel.reshape(-1,1)))
    return x_new

if __name__ == '__main__':
    # NN variables
    input_size = 96
    hidden_size = 256
    ConvSizeOut = 16    #16
    ConvSize = 3
    num_states = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define the network
    model = FC4L256Np05_CNN1L16N_SBP(input_size, hidden_size, ConvSize, ConvSizeOut, num_states).to(device)
    print('Model:', model)
