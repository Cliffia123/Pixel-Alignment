from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("/home/caoxl/BB/Gabor-CNN-Pytorch/")
from gcn.layers import GConv

class GCN(nn.Module):
    def __init__(self, channel=4):
        super(GCN, self).__init__()
        self.channel = channel

        # self.model = nn.Sequential(
        #     GConv(1, 10, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True),
        #     nn.BatchNorm2d(10*channel),
        #     nn.ReLU(inplace=True),
        #
        #     GConv(10, 20, 5, padding=2, stride=1, M=channel, nScale=2, bias=False),
        #     nn.BatchNorm2d(20*channel),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2,2),
        #
        #     GConv(20, 40, 5, padding=0, stride=1, M=channel, nScale=3, bias=False),
        #     nn.BatchNorm2d(40*channel),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2,2),
        #
        #     GConv(40, 80, 5, padding=0, stride=1, M=channel, nScale=4, bias=False),
        #     nn.BatchNorm2d(80*channel),
        #     nn.ReLU(inplace=True),
        # )
        #gabor: [M, nScale,kennel]
        self.conv1 = GConv(1, 10, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True)
        self.bn1 = nn.BatchNorm2d(10*channel)

        self.conv2 = GConv(10, 20, 5, padding=2, stride=1, M=channel, nScale=2, bias=False)
        self.bn2 = nn.BatchNorm2d(20*channel)
        self.maxpool_2 = nn.MaxPool2d(2,2)

        self.conv3 = GConv(20, 40, 5, padding=0, stride=1, M=channel, nScale=3, bias=False)
        self.bn3 = nn.BatchNorm2d(40*channel)
        self.maxpool_3 = nn.MaxPool2d(2,2)

        self.conv4 = GConv(40, 80, 5, padding=0, stride=1, M=channel, nScale=4, bias=False)
        self.bn4 = nn.BatchNorm2d(80*channel)

        self.fc1 = nn.Linear(80, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # x = self.model(x)   #torch.Size([128, 320, 1, 1])

        # x = x.view(-1, 80, self.channel)
        # x = torch.max(x, 2)[0]
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # return x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True) #[1,40,28,28]
        x = self.maxpool_2(F.relu(self.bn2(self.conv2(x)), inplace=True)) #[1,80,14,14]

        x = self.maxpool_3(F.relu(self.bn3(self.conv3(x)), inplace=True))#[1,160,5,5]

        x = F.relu(self.bn4(self.conv4(x)), inplace=True)#[1,320,1,1]


        x = x.view(-1, 80, self.channel) #一张图像4个维度[1,80,4]
        # print("x.shape",x.shape)

        x = torch.max(x, 2)[0]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_network_fn(name):
    networks_zoo = {
    'gcn': GCN(channel=4),
    }
    if name is '':
        raise ValueError('Specify the network to train. All networks available:{}'.format(networks_zoo.keys()))
    elif name not in networks_zoo:
        raise ValueError('Name of network unknown {}. All networks available:{}'.format(name, networks_zoo.keys()))
    return networks_zoo[name]