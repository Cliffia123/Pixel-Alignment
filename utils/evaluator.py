import torch.nn as nn
from torchvision import models
from torch.utils.model_zoo import load_url
import torch
import torch.nn.functional as F


class Evaluator(nn.Module):
    def __init__(self, num_classes=200):
        super(Evaluator, self).__init__()


        self.model = models.vgg16(pretrained=True)
        if num_classes != 1000:
            self.model.classifier[6] = nn.Linear(4096, num_classes)
    def forward(self, x):
        # x = self.conv1_1(x)
        # x = self.relu1_1(x)
        #
        # x = self.conv1_2(x)
        # x = self.relu1_2(x)
        #
        # x = self.pool1(x)
        #
        # # below code --- don't touch any more
        #
        # x = self.conv2_1(x)
        # x = self.relu2_1(x)
        #
        # x = self.conv2_2(x)
        # x = self.relu2_2(x)
        # x = self.pool2(x)
        #
        # x = self.conv3_1(x)
        # x = self.relu3_1(x)
        # x = self.conv3_2(x)
        # x = self.relu3_2(x)
        # x = self.conv3_3(x)
        # x = self.relu3_3(x)
        # x = self.pool3(x)
        #
        # x = self.conv4_1(x)
        # x = self.relu4_1(x)
        # x = self.conv4_2(x)
        # x = self.relu4_2(x)
        # x = self.conv4_3(x)
        # x = self.relu4_3(x)
        # x = self.pool4(x)
        #
        # x = self.conv5_1(x)
        # x = self.relu5_1(x)
        # x = self.conv5_2(x)
        # x = self.relu5_2(x)
        # x = self.conv5_3(x)
        # x = self.relu5_3(x)
        #
        # x = self.conv6_1(x)
        # x = self.relu6_1(x)
        # x = self.fc1(x)
        # x = self.relu6_2(x)
        # x = self.avgpool(x).view(x.size(0),-1)
        # score = self.fc2(x)
        # score = self.model(x)
        # self.feature = self.model.features
        # # self.classifier = self.model.classifier[:5]
        # # print(self.feature,self.classifier)
        # # x = self.model(x)  #
        # # x = self.conv6(x)
        # # feature_map = self.relu(x)
        # # x = self.avgpool(x).view(x.size(0), -1)
        # # score = self.fc(x)
        # #
        # # _, target = score.topk(1, 1, True, True)
        # # fc_weight = self.fc.weight.squeeze()
        # # target = target.squeeze()
        # # cam_weight = fc_weight[target]  # 32 4096
        # # batch, channel, _, _ = feature_map.size()
        # # cam_weight = cam_weight.view(batch, channel, 1, 1).expand_as(feature_map)
        # # cams = (cam_weight * feature_map)  # all are [32,200, 7,7]
        # # cams = cams.mean(1).unsqueeze(1)
        # # cams = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
        # cam = self.feature(x)
        # cam = cam.mean(1).unsqueeze(1)
        # cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

    # def classifier(self, x):
    #     x = self.model.avgpool(x).view(x.size(0),-1)
    #     return self.model.classifier(x)

# def remove_layer(state_dict, keyword):
#     keys = [key for key in state_dict.keys()]
#     for key in keys:
#         if keyword in key:
#             state_dict.pop(key)
#     return state_dict
#
# def make_layers(cfg, batch_norm=False):
#     # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         # MaxPool2d layers in ACoL
#         elif v == 'M1':
#             layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
#         elif v == 'M2':
#             layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=False)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
# cfgs = {
#     'ACoL': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512, 512, 'M2', 512, 512, 512, 'M2'],
#     'ACoL_1': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512, 512, 'M2', 512, 512, 'M2'],
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'D_1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
# }
#
if __name__ == '__main__':
    model = Evaluator(200)
    # model_info(modpythel)
    x = torch.randn(32, 512, 7, 7)
    out= model.classifier(x)
    print(out.shape)