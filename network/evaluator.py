import torch.nn as nn
from torchvision import models
from torch.utils.model_zoo import load_url
import torch
import torch.nn.functional as F


class Evaluator(nn.Module):
    def __init__(self, num_classes=1000):
        super(Evaluator, self).__init__()
        self.model = models.vgg16(pretrained=True)
        if num_classes != 1000:
            self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        self.features =  self.model.features(x)
        return self.model(x)

    def get_fused_cam(self):
        activation_maps = self.features
        cam = torch.sum(activation_maps, dim=1).unsqueeze(1)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        # cam = F.interpolate(self.unet_activate, size=(224, 224), mode='bilinear', align_corners=False)
        return cam

def normalize_tensor(x):
    map_size = x.size()  # 30,1,7,7  20,512,1,1
    aggregated = x.view(map_size[0], map_size[1], -1)  #  20,512,1,1
    minimum, _ = torch.min(aggregated, dim=1, keepdim=True)
    maximum, _ = torch.max(aggregated, dim=1, keepdim=True)
    normalized = torch.div(aggregated - minimum, maximum - minimum)
    normalized = normalized.view(map_size)
    return normalized

if __name__ == '__main__':
    eva = Evaluator(200)
    x = torch.rand(32,3,224,224)
    out,put = eva(x)
    print(put.shape)