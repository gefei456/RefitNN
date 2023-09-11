#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         15:59
# @Author:      WGF
# @File:        RefitNN.py
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
import torchvision.datasets as dset
import torchvision.transforms as T
#import Libraries
import pdb

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
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='tanh')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='tanh')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='tanh')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='tanh')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='tanh')
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
