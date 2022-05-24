import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
import cmapy
from torchvision.utils import make_grid
from torchvision import models
import torch.nn.functional as F
from utils.cam.layercam import *
from utils.evaluator import *
import torch.nn as nn
from utils.cam.utils import *
def get_nor_layercam(activations,gradients):
    with torch.no_grad():
        activation_maps = activations * F.relu(gradients)
        cam = torch.sum(activation_maps, dim=1).unsqueeze(1)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
        cam = norm_cam.mean(1).unsqueeze(1)
        mask = cam.new_ones((gradients.size(0), 1, 224, 224))
        pos = torch.lt(cam, 0.07)
        mask[pos.data] = 0.  # b,1,224,224
    return cam

# def get_mask(map_set):
#
#     map = torch.cat((map_set[0], map_set[1]), dim=1)
#     map = torch.cat((map, map_set[2]), dim=1)
#     map = torch.cat((map, map_set[3]), dim=1)
#     cam = map.mean(1).unsqueeze(1)
#     mask = cam.new_ones((input_.size(0), 1, 224, 224))
#     pos = torch.lt(cam, 0.12)
#     mask[pos.data] = 0.  # b,1,224,224
#     return mask


def get_layerCam(input_, vgg):
    map_set = []
    ids = [30]
    for i in range(len(ids)):
        layer_name = 'features_' + str(i)
        vgg_model_dict = dict(type='vgg', arch=vgg, layer_name=layer_name, input_size=(224, 224))
        vgg_layercam = LayerCAM(vgg_model_dict)
        layercam_map = vgg_layercam(input_)
        map_set.append(layercam_map)
    map = map_set[0]
    # map = torch.cat((map_set[0],map_set[1]), dim=1)
    # map = torch.cat((map,map_set[2]), dim=1)
    # map = torch.cat((map,map_set[3]), dim=1)
    cam = map.mean(1).unsqueeze(1)
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
    # pos = torch.lt(cam, 0.05)
    # cam[pos.data] = 0.  # b,1,224,224

    mask = torch.zeros((input_.size(0), 1, 224, 224))

    pos = torch.lt(cam, 0.003)
    mask[pos.data] = 1. #b,1,224,224

    pos = torch.ge(cam, 0.06)
    mask[pos.data] = 1.  # b,1,224,224

    # for i in range(input_.size(0)):
    #     mask = cam[i].unsqueeze(0)
    #     print(mask.shape)
    #     in_put = input_[i].unsqueeze(0).cpu().detach()
    #     basic_visualize(in_put.cpu().detach(), mask.type(torch.FloatTensor).cpu(),save_path='./re/stage_{}.png'.format(i))

    return mask

from network.loss import _LossExtendedLB

class MaxSizePositiveFcams(nn.Module):
    def __init__(self):
        super(MaxSizePositiveFcams, self).__init__()

        self.elb = _LossExtendedLB(init_t=1., max_t=10., mulcoef=1.01)

    def forward(self, fcams_n=None):

        fcams_n = fcams_n / float(fcams_n.shape[2] * fcams_n.shape[3])
        fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams_n.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)
        print(loss)
def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.
    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.
    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0



if __name__ == '__main__':

    from PIL import Image
    input_image_1 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0098_621.jpg').convert('RGB')
    input_image_2 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0059_8079.jpg').convert('RGB')
    input_image_3 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0007_111029.jpg').convert('RGB')
    input_image_4 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0126_110959.jpg').convert('RGB')
    # input_image_5 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0130_110985.jpg').convert('RGB')
    input_image_5 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/163.Cape_May_Warbler/Cape_May_Warbler_0008_163062.jpg').convert('RGB')
    #

    seeds = torch.zeros((5, 3, 224, 224), requires_grad=False)
    input_1 = apply_transforms(input_image_1)
    input_2 = apply_transforms(input_image_2)
    input_3 = apply_transforms(input_image_3)
    input_4 = apply_transforms(input_image_4)
    input_5 = apply_transforms(input_image_5)
    seeds[0] = input_1
    seeds[1] = input_2
    seeds[2] = input_3
    seeds[3] = input_4
    seeds[4] = input_5
    if torch.cuda.is_available():
      input_ = seeds.cuda()
    vgg = models.vgg16(pretrained=True).eval().cuda()
    # cls_checkpoint = 'train_log/cxz/evaluator/evaluator.pth.tar'
    # vgg.load_state_dict(torch.load(cls_checkpoint, map_location='cpu')['state_dict'])
    # import torch.nn.functional as F
    #
    # score = vgg(input_)
    # features = vgg.features[:30](input_)
    # cam = torch.sum(features, dim=1).unsqueeze(1)
    # cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    # cam_min, cam_max = cam.min(), cam.max()
    # cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data

    cam = get_layerCam(input_, vgg)

    # IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
    # IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
    # image_mean = torch.reshape(torch.tensor(IMAGE_MEAN_VALUE), (1, 3, 1, 1))
    # image_std = torch.reshape(torch.tensor(IMAGE_STD_VALUE), (1, 3, 1, 1))
    # image_ = seeds.clone().detach().cpu() * image_mean + image_std
    # image_ = image_.numpy().transpose(0, 2, 3, 1)
    #
    # image_ = image_[:, :, :, ::-1] * 255
    # #
    for i in range(5):
        mask = cam[i].unsqueeze(0) #1 1 224 224
        print(mask.shape)
        in_put = input_[i].unsqueeze(0).cpu().detach()
        # heatmap = intensity_to_rgb(mask, normalize=True).astype('uint8')
        # heatmap_BGR = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        #
        # blend = in_put * 0.4 + heatmap_BGR * 1
        # cv2.imwrite(os.path.join('/home/caoxl/BB/Boundary-en/re', str(i)+'.jpg'), blend)

        basic_visualize(in_put.cpu().detach(), mask.type(torch.FloatTensor).cpu(),save_path='./re/stage_{}.png'.format(i))

    import albumentations as A
    # from albumentations.pytorch import ToTensorV2
    # import cv2
    #
    # transform = A.Compose([
    #     A.Normalize(),
    #     A.Resize(256, 256),
    #     A.RandomCrop(224, 224),
    #     A.HorizontalFlip(p=0.5),
    #     ToTensorV2()
    # ])
    # image = cv2.imread('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0063_111460.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = np.zeros((256,256,1))
    # pair = transform(image=image, mask=mask)
    # image, mask = pair['image'], pair['mask']
    # fore, gaus, back = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
    # mask = torch.maximum(fore > 0.25, gaus > 0.25)
    # weight = torch.maximum(mask, fore < 0.01)
    # print(weight.shape)
    # x = torch.rand(8,1,7,7)
    # a = torch.rand(3,1,2,2)
    #
    # print(a[:, 0, :, :].shape)
