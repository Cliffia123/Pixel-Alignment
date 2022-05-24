
USE_PYTORCH_LIGHTNING=False

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.utils.model_zoo import load_url
import torch.utils.model_zoo as model_zoo

import torchvision.models as models
import numpy as np
from utils.ddt.ddt_func import *
from torch.autograd import Variable
from utils.ddt.pca_project import *
from network.vgg16_cel import *
from network.vgg16_eil import *
from network.vgg16_merge import *
from network.crf.dense_crf_loss import *
from network.unet import *

__all__ = [
    'VGG', 'vgg16_acol',
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}
layer_mapping_vgg = OrderedDict([('features.0.weight', 'conv1_1.weight'), ('features.0.bias', 'conv1_1.bias'), ('features.2.weight', 'conv1_2.weight'), ('features.2.bias', 'conv1_2.bias'), ('features.5.weight', 'conv2_1.weight'), ('features.5.bias', 'conv2_1.bias'), ('features.7.weight', 'conv2_2.weight'), ('features.7.bias', 'conv2_2.bias'), ('features.10.weight', 'conv3_1.weight'), ('features.10.bias', 'conv3_1.bias'), ('features.12.weight', 'conv3_2.weight'), ('features.12.bias', 'conv3_2.bias'), ('features.14.weight', 'conv3_3.weight'), (
    'features.14.bias', 'conv3_3.bias'), ('features.17.weight', 'conv4_1.weight'), ('features.17.bias', 'conv4_1.bias'), ('features.19.weight', 'conv4_2.weight'), ('features.19.bias', 'conv4_2.bias'), ('features.21.weight', 'conv4_3.weight'), ('features.21.bias', 'conv4_3.bias'), ('features.24.weight', 'conv5_1.weight'), ('features.24.bias', 'conv5_1.bias'), ('features.26.weight', 'conv5_2.weight'), ('features.26.bias', 'conv5_2.bias'), ('features.28.weight', 'conv5_3.weight'), ('features.28.bias', 'conv5_3.bias')])

class VGG(nn.Module):

    def __init__(self, num_classes=200, init_weights=True):
        super(VGG, self).__init__()
        self.numclasses = num_classes
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0, bias=False)
        self.att_conv3 = nn.Conv2d(200, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)

        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)

        # 64 x 224 x 224
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)

        # 64 x 112 x 112
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128 x 112 x 112
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)

        # 128 x 112 x 112
        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

        # 128 x 56 x 56
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256 x 56 x 56
        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv3_3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)

        # 256 x 28 x 28
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512 x 28 x 28
        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)

        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)

        self.conv4_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)

        #  512 x 14 x 14
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512 x 14 x 14
        self.conv5_1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)

        self.conv5_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv5_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = self.make_classifier()
        # self.fc51 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        # self.bn51 = nn.BatchNorm2d(128)
        # self.fc52 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        #
        # self.fc41 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        # self.bn41 = nn.BatchNorm2d(128)
        # self.fc42 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        #
        # self.fc31 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        # self.bn31 = nn.BatchNorm2d(128)
        # self.fc32 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        #
        # self.linear5 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)
        # self.bn5 = nn.BatchNorm2d(1024)
        #
        # self.linear4 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)
        # self.bn4 = nn.BatchNorm2d(1024)
        #
        # self.linear3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(1024)
        #
        # self.linear = nn.Conv2d(1024, 200, kernel_size=1, stride=1, padding=0)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, **kwargs):
        x = self.conv1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)

        # below code --- don't touch any more

        x = self.conv2_1(x)
        x = self.relu2_1(x)

        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)

        x = self.conv3_2(x)
        x = self.relu3_2(x)

        x = self.conv3_3(x)
        x = self.relu3_3(x)
        self.out3 = x #512 512
        x = self.pool3(x)

        self.pool3_fea = x
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)

        x = self.conv4_3(x)
        x = self.relu4_3(x)
        self.out4 = x.clone() #512 512 28 28
        x = self.pool4(x)

        # self.x = x
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        self.x_51 = x

        x = self.conv5_2(x)
        x = self.relu5_2(x)
        self.x_52 = x.clone() #512 512

        x = self.conv5_3(x)
        x = self.relu5_3(x)
        self.x_53 = x
        x = x.view(x.size(0), -1)
        # x = self.avgpool(x).view(x.size(0), -1) #16,200
        # classify = self.classifier(x)
        self.score = x
        return self.score

    def get_fused_cam(self):

        batch, channel, _, _ = self.x_local.size()
        fc_weight = self.fc.weight.squeeze()

        if target is None:
            _, target = self.score.topk(1, 1, True, True)

        target = target.squeeze()

        # get fc weight (num_classes x channel) -> (batch x channel)
        cam_weight = fc_weight[target]  #32 1024
        # get final cam with weighted sum of feature map and weights
        # (batch x channel x h x w) * ( batch x channel)
        cam_weight = cam_weight.view(
            batch, channel, 1, 1).expand_as(self.feature_map)
        cam = (cam_weight * self.feature_map)   #all are [32,1024, 14,14]
        cam = cam.mean(1)
        cam = self.feature_map.mean(1)

        #caoxz add 2021.03
        cam = F.interpolate(self.x_52, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.mean(1)
        cam = cam.unsqueeze(1)
        return cam

    def get_loss(self, args, target):
        loss_cls = self.CrossEntropyLoss(self.score_cls, target)
        # loss_fg = self.CrossEntropyLoss(self.score_fg, target)

        # loss_unet = self.CrossEntropyLoss(self.score_unet, target)

        # loss_bg = torch.sum(self.score_bg/ (self.score_out5.detach()+0.00035))/(self.score_out5.shape[0]*self.score_out5.shape[1]) #16,200
        #
        # loss_ac = torch.sum(self.x_local)/(self.x_local.shape[0]*self.x_local.shape[2]*self.x_local.shape[3])
        #
        # core = loss_cls+loss_bg+0.7*loss_ac
        return loss_cls


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_classifier(self):
        return nn.Sequential(
            # nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(1024, 200, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(inplace=True),
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.numclasses),
        )

def normalize_tensor(x):
    map_size = x.size()  # 30,1,7,7  20,512,1,1
    aggregated = x.view(map_size[0], map_size[1], -1)  #  20,512,1,1
    minimum, _ = torch.min(aggregated, dim=1, keepdim=True)
    maximum, _ = torch.max(aggregated, dim=1, keepdim=True)
    normalized = torch.div(aggregated - minimum, maximum - minimum)  # div除法
    normalized = normalized.view(map_size)
    # atten_shape = x.size()
    # batch_mins, _ = torch.min(x.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    # batch_maxs, _ = torch.max(x.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
    # atten_normed = torch.div(x.view(atten_shape[0:-2] + (-1,)) - batch_mins, batch_maxs - batch_mins + 1e-10)
    # atten_normed = atten_normed.view(atten_shape)
    return normalized


def make_localization():
    #分割网络
    return nn.Sequential(
        nn.Conv2d(512, 200, kernel_size=3, padding=1),  ## num_classes
        nn.Sigmoid(),
    )
def make_layers(cfg, batch_norm=False):
    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # MaxPool2d layers in ACoL
        elif v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'ACoL': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512, 512, 'M2', 512, 512, 512, 'M2'],
    'ACoL_1': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512, 512, 'M2', 512, 512, 'M2'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

def _vgg_ori(pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs['D_1']), **kwargs)
    if pretrained:
        state_dict = load_url(model_urls['vgg16'])
        state_dict = remove_layer(state_dict, 'classifier.')
        model.load_state_dict(state_dict, strict=False)
    return model

def _vgg_val(pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = models.vgg16(pretrained=True)
    if torch.cuda.is_available():
        model.cuda()
    if pretrained:
        state_dict = load_url(model_urls['vgg16'])
        model.load_state_dict(state_dict, strict=True)
    return model


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict

def _vgg(pretrained=False, progress=True, **kwargs):
    kwargs['init_weights'] = True
    model = VGG( **kwargs)
    model_dict = model.state_dict()
    if pretrained:
        print("coming")
        pretrained_dict = models.vgg16(pretrained=True).state_dict()
        pretrained_dict = remove_layer(pretrained_dict, 'classifier.')

        for pretrained_k in pretrained_dict:
            if pretrained_k not in layer_mapping_vgg.keys():
                continue

            my_k = layer_mapping_vgg[pretrained_k]
            if my_k not in model_dict.keys():
                my_k = "module."+my_k
            if my_k not in model_dict.keys():
                raise Exception("Try to load not exist layer...")
            model_dict[my_k] = pretrained_dict[pretrained_k]
    model.load_state_dict(model_dict)
    return model