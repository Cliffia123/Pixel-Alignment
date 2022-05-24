import os
import sys
import torch
from collections import OrderedDict

import torch.nn.functional as F
import torch.nn as nn
sys.path.append(os.getcwd())
from network.unet import UNet
# from network.vgg16_acol import VGG
# from network import vgg16_acol
from network.loss import AreaLoss,BackgroundLoss,DenseCRFLoss,WeightedEntropyLoss
from network.erasing import AttentiveErasing
from network.vgg16_merge import Merge
from torchvision import models
from network.evaluator import Evaluator
from network.core import heads
from utils.cam.layercam import *
class CoarseModel(nn.Module):
    def __init__(self, num_classes=200):
        super(CoarseModel, self).__init__()

        self.att_layer4 = UNet(64, num_classes, c_base=64, activation='leakyrelu')

        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
                                   bias=False)
        self.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
                                   bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(56)
        # self.classifier = make_classifier(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        model = models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool

        if num_classes == 200:
            # self.classifier = models.vgg16(pretrained=True).classifier
            # self.model.classifier[6] = nn.Linear(4096, num_classes)
            self.fc = nn.Linear(2048, num_classes)
        else:
            model = models.resnet50(pretrained=False)
            self.layer4 = model.layer4
            self.fc = model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        sx = self.att_layer4(x)
        att = torch.sigmoid(self.bn_att3(self.att_conv3(sx)))
        self.unet_features = sx

        # att branch
        sx = self.att_conv2(sx)
        sx = self.att_gap(sx)
        sx = sx.view(sx.size(0), -1)
        self.score_unet = sx

        # main branch
        mx = x * att
        mx = self.layer1(mx)  # 1024 14 14
        mx = self.layer2(mx)  # 2048 7 7
        mx = self.layer3(mx) # 1024 14 14
        mx = self.layer4(mx) # 2048 7 7
        mx = self.avgpool(mx)
        mx = mx.view(mx.size(0), -1)
        self.score = self.fc(mx)
        self.unet_activate = att
        # self.score_unet = sx


        return {'score': self.score,'score_unet':self.score_unet}

    def get_fused_cam(self):
        # cam = self.unet_features.mean(1).unsqueeze(1)
        # cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return self.unet_activate

class FineModel(nn.Module):
    def __init__(self, att_checkpoint, cls_checkpoint, args, max_factor=0.5, num_classes=200):
        super(FineModel, self).__init__()
        model = CoarseModel(num_classes=num_classes)
        # model = CoarseModel_2(num_classes=num_classes)
        self.evaluate = args.evaluate

        # for p in model.parameters():
        #     p.requires_grad = False
        # for p in model.att_layer4.up_1.parameters():
        #     p.requires_grad = True
        # for p in model.att_layer4.up_2.parameters():
        #     p.requires_grad = True
        # for p in model.att_layer4.up_3.parameters():
        #     p.requires_grad = True
        # # for p in model.att_layer4.down_1.parameters():
        # #     p.requires_grad = True
        # # for p in model.att_layer4.down_2.parameters():
        # #     p.requires_grad = True
        # # for p in model.att_layer4.down_3.parameters():
        # #     p.requires_grad = True
        # for p in model.att_conv3.parameters():
        #     p.requires_grad = True
        # for p in model.bn_att3.parameters():
        #     p.requires_grad = True
        # # # initialization from the coarse model
        model = nn.DataParallel(model)  #model.module
        mod = remove_layer(torch.load(att_checkpoint, map_location='cpu')['state_dict'], 'vgg')
        print("--------------------localization name--------------------")
        model.load_state_dict(mod)
        model = model.module

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.fc = model.fc
        self.att_layer4 = model.att_layer4

        self.att_conv3 = model.att_conv3
        self.bn_att3 = model.bn_att3
        self.att_conv2 = model.att_conv2
        self.att_gap = model.att_gap
        self.avgpool = model.avgpool
        self.erasing = AttentiveErasing(max_factor=max_factor)

        model_eval = Evaluator(num_classes)
        att = torch.load(cls_checkpoint, map_location='cpu')['state_dict']
        model_eval.load_state_dict(att)

        for p in model_eval.parameters():
            p.requires_grad = False
        self.vgg = model_eval
        self.gradients = dict()
        self.activations = dict()

    def forward(self, x):

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        pool = self.maxpool(relu)

        sx = self.att_layer4(pool)
        att = torch.sigmoid(self.bn_att3(self.att_conv3(sx)))
        self.unet_features = sx

        self.unet_activate = att
        if self.evaluate:
            bx = x
        else:
            att_dropout, _ = self.erasing(att)
            bx = x * F.interpolate(att_dropout, size=(x.size(3), x.size(3)))
        self.score = self.vgg(bx)

        return {'score': self.score,'unet_activate':self.unet_activate,
                'activate_features': self.unet_features}

    def get_mask(self):

        # batch, channel, _, _ = self.feature_map.size()
        # _, target = self.score_unet.topk(1, 1, True, True)
        # fc_weight = self.fc.weight.squeeze()
        # target = target.squeeze()
        # cam_weight = fc_weight[target]  # 32 2048
        # cam_weight = cam_weight.view(batch, channel, 1, 1).expand_as(self.feature_map)
        # cam = (cam_weight * self.feature_map)  # all are [32,200, 7,7]
        mask = torch.zeros((self.cam.size(0), 224, 224))
        pos = torch.ge(self.cam, 0.5)
        mask[pos.data] = 1.
        weight = torch.maximum(mask.cuda(), self.cam < 0.07)
        return  weight
    # def rest_fea(self):
    # def erasing_mask(self):
    #     cam = self.feature.mean(1).unsqueeze(1)
    #     cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    #     mask = torch.maximum(cam<0.5, self.unet_activate<0.5)
    #     erasing_att = mask*self.unet_activate
    #     return erasing_att

    def get_fused_cam(self):
        # batch, channel, _, _ = self.feature_map.size()
        # _, target = self.score_fg.topk(1, 1, True, True)
        # fc_weight = self.fc.weight.squeeze()
        # target = target.squeeze()
        #
        # cam_weight = fc_weight[target]  # 32 2048
        #
        # cam_weight = cam_weight.view(batch, channel, 1, 1).expand_as(self.feature_map)
        # cam = (cam_weight * self.feature_map)  # all are [32,200, 7,7]
        #
        # # _, pre_label = self.score.topk(180, 1, True, True)
        # # cam_bg = torch.zeros(batch, 20, 7, 7).cuda()
        # # for i in range(self.feature_map.size(0)):
        # #     for j in range(200):
        # #         idx = 0
        # #         if j not in pre_label:
        # #             cam_bg[i][idx] = self.feature_map[i][j]
        # #             idx+=1
        # cam_ = cam.mean(1).(1)
        # cams = self.cam_feature.mean(1).unsqueeze(1)
        # cam = self.unet_feature[:,1:,:]
        # cam = self.unet_features.mean(1).unsqueeze(1)
        cam = F.interpolate(self.unet_activate, size=(224, 224), mode='bilinear', align_corners=False)
        return cam

    def get_attention(self, attention_map):
        b, _, h, w = attention_map.shape
        mask = attention_map.new_ones((b,1,h,w))
        pos = torch.ge(attention_map, 0.4)
        mask[pos.data] = 0.
        attention = attention_map*mask
        return attention

    def backward_hook(self, module, grad_input, grad_output):
        # if torch.cuda.is_available():
        #   self.gradients['value'] = grad_output[0].cuda(4, non_blocking=True)
        # else:
        self.gradients['value'] = grad_output[0]
        return None

    def forward_hook(self, module, input, output):
        # if torch.cuda.is_available():
        #   self.activations['value'] = output.cuda(4, non_blocking=True)
        # else:
        self.activations['value'] = output
        return None




def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


class CoarseModel_2(nn.Module):
    def __init__(self, num_classes=200):
        super(CoarseModel_2, self).__init__()

        self.att_layer4 = UNet(64, num_classes, c_base=64, activation='leakyrelu')
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
                                   bias=False)

        self.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
                                   bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(56)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        model = models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.avgpool = model.avgpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # # self.merge = Merge()
        #
        #
        if num_classes == 200:
            self.fc = nn.Linear(2048, num_classes)
        # else:
        #     model = models.resnet50(pretrained=False)
        #     # self.layer4 = model.layer4
        #     self.fc = model.fc

def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict

# def get_pseudo_binary_mask(x):
#     """
#     Compute a mask by applying a sigmoid function.
#     The mask is not binary but pseudo-binary (values are close to 0/1).
#
#     :param x: tensor of size (batch_size, 1, h, w), cont    ain the feature
#      map representing the mask.
#     :return: tensor, mask. with size (nbr_batch, 1, h, w).
#     """
#     # wrong: x.min() .max() operates over the entire tensor.
#     # it should be done over each sample.
#     x = (x - x.min()) / (x.max() - x.min())
#     return torch.sigmoid(self.w * (x - self.sigma))
def normalize_tensor(x):
    map_size = x.size()  # 30,1,7,7  20,512,1,1
    aggregated = x.view(map_size[0], map_size[1], -1)  #  20,512,1,1
    minimum, _ = torch.min(aggregated, dim=1, keepdim=True)
    maximum, _ = torch.max(aggregated, dim=1, keepdim=True)
    normalized = torch.div(aggregated - minimum, maximum - minimum)
    normalized = normalized.view(map_size)
    return normalized

