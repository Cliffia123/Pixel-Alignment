
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

    def __init__(self, num_classes=200):
        super(VGG, self).__init__()
        self.numclasses = num_classes
        self.model = models.vgg16(pretrained=True)
        if num_classes != 1000:
            self.model.classifier[6] = nn.Linear(4096, num_classes)
    def forward(self, x):

        return self.model(x)

    # def get_fused_cam(self):
    #     #caoxz add 2021.03
    #     cam = F.interpolate(self.x_52, size=(224, 224), mode='bilinear', align_corners=False)
    #     cam = cam.mean(1)
    #     cam = cam.unsqueeze(1)
    #     return cam