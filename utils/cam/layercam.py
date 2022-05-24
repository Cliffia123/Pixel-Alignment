import torch
import torch.nn.functional as F
from utils.cam.basecam import *
from torchvision import models


class LayerCAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        # predication on raw input
        logit = self.model_arch(input).cuda()
        predicted_class = logit.max(1)[-1]
        # if torch.cuda.is_available():
        #     predicted_class = predicted_class.cuda(4, non_blocking=True)
        #     logit = logit.cuda(4, non_blocking=True)
        one_hot_output = torch.FloatTensor(logit.size(0), logit.size()[-1]).zero_()
        for i in range(logit.size(0)):
            one_hot_output[i][predicted_class] = 1
        one_hot_output = one_hot_output.cuda()

        # Zero grads
        # self.model_arch.zero_grad()
        # Backward pass with specified target
        logit.backward(gradient=one_hot_output, retain_graph=True)
        activations = self.activations['value'].clone().detach()
        gradients = self.gradients['value'].clone().detach()

        with torch.no_grad():
            activation_maps = activations * F.relu(gradients)
            # activation_maps = activations
            cam = torch.sum(activation_maps, dim=1).unsqueeze(1)
            cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
            cam_min, cam_max = cam.min(), cam.max()
            norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
        return norm_cam

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)