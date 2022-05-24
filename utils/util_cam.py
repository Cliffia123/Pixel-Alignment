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
# from evaluator import *
import torch.nn as nn

def get_gussian(cam):
    # gaus_tensor = torch.zeros((cam.size(0), 224, 224))
    # cam_ori = cam.squeeze(1)
    # cam = cam.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)
    #
    # for i in range(gaus_tensor.size(0)):
    #     pred = cam[i]
    #     pred = cv2.resize(pred, (224, 224), interpolation=cv2.INTER_LINEAR)
    #     weights = pred.copy()
    #     weights[np.where(pred < 0.1)] = 0
    #     weights[np.where(pred > 0.5)] = 0
    #     contours = cv2.findContours(np.uint8(weights > 0) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    #
    #     gaus = 0
    #     for i in range(len(contours)):
    #         if cv2.contourArea(contours[i]) < 20:
    #             continue
    #         weight = np.zeros((224, 224, 3))
    #         weight = cv2.drawContours(weight, contours, i, color=(1, 1, 1), thickness=-1, lineType=None, hierarchy=None,
    #                                   maxLevel=None, offset=None)
    #         weight = weight[:, :, 0] * weights
    #         weight = weight / weight.sum()
    #         X, Y = np.meshgrid(np.arange(224), np.arange(224))
    #         ux, uy = (weight * X).sum(), (weight * Y).sum()
    #         sx, sy = (weight * (X - ux) ** 2).sum(), (weight * (Y - uy) ** 2).sum()
    #         sxy = (weight * (X - ux) * (Y - uy)).sum()
    #         gaus = np.maximum(gaus, gaussian(X, Y, ux, uy, sx / 10, sy / 10, sxy / 10))
    #
    #         gaus_tensor[i] = torch.from_numpy(gaus)
    # mask = torch.maximum(cam_ori > 0.5, gaus_tensor.cuda() > 0.25)
    pos = torch.le(cam, 0.1)
    mask[pos.data] = 1.

    weight = torch.maximum(mask, cam < 0.05)

    return weight

def gaussian(x, y, ux, uy, sx, sy, sxy, pred=None):
    c   = -1/(2*(1-sxy**2/sx/sy))
    dx  = (x-ux)**2/sx
    dy  = (y-uy)**2/sy
    dxy = (x-ux)*(y-uy)*sxy/sx/sy
    return np.exp(c*(dx-2*dxy+dy))

def get_nor_layercam(activations,gradients):

    with torch.no_grad():
        activation_maps = activations * F.relu(gradients)
        cam =  torch.sum(activation_maps, dim=1).unsqueeze(1)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data

        mask = torch.zeros((gradients.size(0), 1, 224, 224))
        pos = torch.ge(norm_cam, 0.07)
        mask[pos.data] = 1.  # b,1,224,224
        weight = torch.maximum(mask.cuda(), norm_cam<0.03)
        # gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

    return norm_cam, weight

def normal_cam(cam):
    cam_min, cam_max = cam.min(), cam.max()
    norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
    return norm_cam
def get_mask(map_set):

    map = torch.cat((map_set[0], map_set[1]), dim=1)
    map = torch.cat((map, map_set[2]), dim=1)
    map = torch.cat((map, map_set[3]), dim=1)
    cam = map.mean(1).unsqueeze(1)
    mask = cam.new_ones((input_.size(0), 1, 224, 224))
    pos = torch.lt(cam, 0.12)
    mask[pos.data] = 0.  # b,1,224,224
    return mask


def get_layerCam(input_, args):
    map_set = []
    ids = [26, 30]
    vgg = models.vgg16(pretrained=True).eval().cuda(args.gpu)
    for i in range(len(ids)):
        layer_name = 'features_' + str(ids[1])
        vgg_model_dict = dict(type='vgg', arch=vgg, layer_name=layer_name, input_size=(224, 224))
        vgg_layercam = LayerCAM(vgg_model_dict)
        layercam_map = vgg_layercam(input_)
        map_set.append(layercam_map)
    # cam = torch.zeros((input_.size(0), 1, 224, 224)).cuda()
    # for i in range(len(ids)):
        # cam = torch.max(cam, map_set[i])
    cam = torch.cat((map_set[0],map_set[1]), dim=1)
    cam = cam.mean(1).unsqueeze(1)
    mask = torch.zeros((input_.size(0), 1, 224, 224))
    pos = torch.ge(cam, 0.5)
    mask[pos.data] = 1. #b,1,224,224
    #
    for i in range(input_.size(0)):
        mask_ = mask[i].unsqueeze(0)
        print(mask.shape)
        in_put = input_[i].unsqueeze(0).cpu().detach()
        basic_visualize(in_put.cpu().detach(), mask_.type(torch.FloatTensor).cpu(),save_path='./vis/stage_{}.png'.format(i))
    # #
    return mask


def get_cam(model, image=None, args=None, target=None):
    """
        Return CAM tensor which shape is (batch, 1, h, w)
    """
    with torch.no_grad():
        if image is not None:
            _ = model(image)

        # Extract feature map
        if args.distributed:
            heatmap = model.module.get_fused_cam()
        else:
            heatmap = model.get_fused_cam()
        return heatmap

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

def intensity_to_gray(intensity, normalize=True, _sqrt=False):
    assert intensity.ndim == 2

    if _sqrt:
        intensity = np.sqrt(intensity)

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    intensity = np.uint8(255*intensity)
    return intensity

def resize_cam(cam, size=(224, 224)):
    cam = cv2.resize(cam, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam

def blend_cam(image, cam):

    mask = cam.repeat(3, 1, 1).permute(1, 2, 0).cpu().detach().numpy()
    mask = cv2.resize(mask, (224, 224))
    mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    save_img = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
    save_img = save_img.transpose(2, 1, 0)
    save_img = torch.from_numpy(save_img)
    return save_img

def blend_ddt_cam(image_class, image_name, cam, args):

    if args.dataset == "COCO":
        image = cv2.resize(cv2.imread(os.path.join(args.data_root,  image_name)), (224, 224))

    elif args.dataset == "CUB":
        image = cv2.resize(cv2.imread(os.path.join(args.data_root, image_class, image_name)), (224, 224))

    elif args.dataset == "ILSVRC":
        # folder_name, name = image_name.split('/')
        # print(os.path.join(args.data_root, 'val',folder_name, name))
        # image = cv2.resize(cv2.imread(os.path.join(args.data_root, 'train',folder_name, name)), (224, 224))
        image = cv2.resize(cv2.imread(os.path.join(args.data_root, 'test', image_name)), (224, 224))
    mask = cam.repeat(3, 1, 1).permute(1, 2, 0).cpu().detach().numpy()*255.
    mask = cv2.resize(mask, (224, 224))
    mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    save_img = cv2.addWeighted(image, 0.3, mask, 0.8, 0.0)

    return save_img

def find_bbox(scoremap, threshold=0.5, scale=4):
    if isinstance(threshold, list):
        bboxes = []
        for i in range(len(threshold)):
            indices = np.where(scoremap > threshold[i])
            try:
                miny, minx = np.min(indices[0] * scale), np.min(indices[1] * scale)
                maxy, maxx = np.max(indices[0] * scale), np.max(indices[1] * scale)
            except:
                bboxes.append([0, 0, 224, 224])
            else:
                bboxes.append([minx, miny, maxx, maxy])
        return bboxes

    else:
        indices = np.where(scoremap > threshold)
        try:
            miny, minx = np.min(indices[0] * scale), np.min(indices[1] * scale)
            maxy, maxx = np.max(indices[0] * scale), np.max(indices[1] * scale)
        except:
            return [0, 0, 224, 224]
        else:
            return [minx, miny, maxx, maxy]

def generate_bbox(image, cam, gt_bbox, thr_val, args):
    '''
    image: single image with shape (h, w, 3)
    cam: single image with shape (h, w, 1), data type is numpy
    gt_bbox: [x, y, x + w, y + h]
    thr_val: float value (0~1)

    return estimated bounding box, blend image with boxes
    '''
    image_height, image_width, _ = image.shape
    # print("here get image shape",image_height,image_width)
    # print("but input atten shape",cam.shape)

    _gt_bbox = list()
    _gt_bbox.append(max(int(gt_bbox[0]), 0))
    _gt_bbox.append(max(int(gt_bbox[1]), 0))
    _gt_bbox.append(min(int(gt_bbox[2]), image_height - 1))
    _gt_bbox.append(min(int(gt_bbox[3]), image_width))

    cam = cv2.resize(cam, (image_height, image_width),interpolation=cv2.INTER_NEAREST)

    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    heatmap_BGR = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    # heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_cls), cv2.COLORMAP_JET)
    blend = image * 0.4 + heatmap_BGR * 1
    # gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    gray_heatmap = intensity_to_gray(cam, normalize=True)

    thr_val = thr_val * np.max(gray_heatmap)

    _, thr_gray_heatmap = cv2.threshold(gray_heatmap,
                                        int(thr_val), 255,
                                        cv2.THRESH_BINARY)

    if args.bbox_mode == 'classical':

        dt_gray_heatmap = thr_gray_heatmap


        try:
            _, contours, _ = cv2.findContours(dt_gray_heatmap,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours, _ = cv2.findContours(dt_gray_heatmap,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

        _img_bbox = (image.copy()).astype('uint8')

        blend_bbox = blend.copy()
        cv2.rectangle(blend_bbox,
                      (_gt_bbox[0], _gt_bbox[1]),
                      (_gt_bbox[2], _gt_bbox[3]),
                      (0, 0, 255), 2)

        # may be here we can try another method to do
        # TODO
        # threshold all the box and then merge it
        # and then rank it,
        # finally merge it
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            estimated_bbox = [x, y, x + w, y + h]
            cv2.rectangle(blend_bbox,
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0), 2)
        else:
            estimated_bbox = [0, 0, 1, 1]
        # cam = torch.from_numpy(cam)
        return estimated_bbox, blend_bbox, blend

    elif args.bbox_mode == 'DANet':  # mode is union
        def extract_bbox_from_map(boolen_map):
            assert boolen_map.ndim == 2, 'Invalid input shape'
            rows = np.any(boolen_map, axis=1)
            cols = np.any(boolen_map, axis=0)
            if rows.max() == False or cols.max() == False:
                return 0, 0, 0, 0
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            # here we modify a box to a list
            return [xmin, ymin, xmax, ymax]

        # thr_gray_map is a gray map
        estimated_bbox = extract_bbox_from_map(thr_gray_heatmap)
        blend_bbox = blend.copy()
        cv2.rectangle(blend_bbox,
                      (_gt_bbox[0], _gt_bbox[1]),
                      (_gt_bbox[2], _gt_bbox[3]),
                      (0, 0, 255), 2)
        cv2.rectangle(blend_bbox,
                      (estimated_bbox[0], estimated_bbox[1]),
                      (estimated_bbox[2], estimated_bbox[3]),
                      (0, 255, 0), 2)
        # cam = torch.from_numpy(cam)
        return estimated_bbox, blend_bbox, gray_heatmap

def visualization(images, attmaps, image_name, image_ids, gt_bbox):
    _, c, h, w = images.shape

    for i in range(images.shape[0]):
        box = gt_bbox[int(image_ids[i])][0]

        _gt_bbox = list()
        _gt_bbox.append(max(int(box[0]), 0))
        _gt_bbox.append(max(int(box[1]), 0))
        _gt_bbox.append(min(int(box[2]), h - 1))
        _gt_bbox.append(min(int(box[3]), w))

        attmap = attmaps[i]
        attmap = attmap / np.max(attmap)
        attmap = np.uint8(attmap * 255)

        colormap = cv2.applyColorMap(cv2.resize(attmap, (w, h)), cmapy.cmap('seismic'))

        grid = make_grid(images[i].unsqueeze(0), nrow=1, padding=0, pad_value=0,
                         normalize=True, range=None)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        image = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[..., ::-1]

        cam = colormap + 0.5 * image
        cam = cam / np.max(cam)
        cam = np.uint8(cam * 255).copy()

        cv2.rectangle(cam,(_gt_bbox[0], _gt_bbox[1]), (_gt_bbox[2], _gt_bbox[3]), (0, 0, 255), 2)

        box = find_bbox(cam)
        cv2.rectangle(cam, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        saving_folder = os.path.join('train_log/cxz/','res')
        if not os.path.isdir(saving_folder):
            os.makedirs(saving_folder)
        cv2.imwrite(f'train_log/cxz/res/{image_name[i]}_att.jpg', cam)

def load_bbox(args):
    """ Load bounding box information """
    origin_bbox = {}
    image_sizes = {}
    resized_bbox = {}

    dataset_path = args.data_list
    resize_size = args.resize_size
    crop_size = args.crop_size
    if args.dataset == 'CUB':
        with open(os.path.join(dataset_path, 'bounding_boxes.txt')) as f:
            for each_line in f:
                file_info = each_line.strip().split()
                image_id = int(file_info[0])

                boxes = map(float, file_info[1:])

                origin_bbox[image_id] = list(boxes)

        with open(os.path.join(dataset_path, 'sizes.txt')) as f:
            for each_line in f:
                file_info = each_line.strip().split()
                image_id = int(file_info[0])
                image_width, image_height = map(float, file_info[1:])

                image_sizes[image_id] = [image_width, image_height]

        if args.VAL_CROP:
            shift_size = (resize_size - crop_size) // 2
        else:
            resize_size = crop_size
            shift_size = 0
        for i in origin_bbox.keys():
            num_boxes = len(origin_bbox[i]) // 4
            for j in range(num_boxes):
                x, y, bbox_width, bbox_height = origin_bbox[i][j:j+4]
                image_width, image_height = image_sizes[i]
                left_bottom_x = int(max(x / image_width * resize_size - shift_size, 0))
                left_bottom_y = int(max(y / image_height * resize_size - shift_size, 0))

                right_top_x = int(min((x + bbox_width) / image_width * resize_size - shift_size, crop_size - 1))
                right_top_y = int(min((y + bbox_height) / image_height * resize_size - shift_size, crop_size - 1))
                resized_bbox[i] = [[left_bottom_x, left_bottom_y, right_top_x, right_top_y]]

    elif args.dataset == 'ILSVRC':
        with open(os.path.join(dataset_path, 'gt_ImageNet.pickle'), 'rb') as f:
            info_imagenet = pickle.load(f)

        origin_bbox = info_imagenet['gt_bboxes']
        image_sizes = info_imagenet['image_sizes']

        if args.VAL_CROP:
            shift_size = (resize_size - crop_size) // 2
        else:
            resize_size = crop_size
            shift_size = 0
        for key in origin_bbox.keys():

            image_height, image_width = image_sizes[key]
            resized_bbox[key] = list()
            x_min, y_min, x_max, y_max = origin_bbox[key][0]
            left_bottom_x = int(max(x_min / image_width * resize_size - shift_size, 0))
            left_bottom_y = int(max(y_min / image_height * resize_size - shift_size, 0))
            right_top_x = int(min(x_max / image_width * resize_size - shift_size, crop_size - 1))
            right_top_y = int(min(y_max / image_height * resize_size - shift_size, crop_size - 1))

            resized_bbox[key].append([left_bottom_x, left_bottom_y, right_top_x, right_top_y])
    else:
        raise Exception("No dataset named {}".format(args.dataset))
    return resized_bbox


def get_bboxes(cam, cam_thr=0.4):
    """
    image: single image with shape (h, w, 3)
    cam: single image with shape (h, w, 1)
    gt_bbox: [x, y, x + w, y + h]
    thr_val: float value (0~1)
    return estimated bounding box, blend image with boxes
    """
    cam_ = cam.permute(1, 2, 0).cpu().numpy()
    cam_ = cv2.resize(cam_, (224, 224))
    cam = (cam_ * 255.).astype(np.uint8)

    map_thr = cam_thr * np.max(cam)
    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_BINARY)

    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 0, 0]

    return estimated_bbox

# def find_bbox(scoremap, threshold=0.5, scale=4):
#     if isinstance(threshold, list):
#         bboxes = []
#         for i in range(len(threshold)):
#             indices = np.where(scoremap > threshold[i])
#             try:
#                 miny, minx = np.min(indices[0] * scale), np.min(indices[1] * scale)
#                 maxy, maxx = np.max(indices[0] * scale), np.max(indices[1] * scale)
#             except:
#                 bboxes.append([0, 0, 224, 224])
#             else:
#                 bboxes.append([minx, miny, maxx, maxy])
#         return bboxes
#
#     else:
#         indices = np.where(scoremap > threshold)
#         try:
#             miny, minx = np.min(indices[0] * scale), np.min(indices[1] * scale)
#             maxy, maxx = np.max(indices[0] * scale), np.max(indices[1] * scale)
#         except:
#             return [0, 0, 224, 224]
#         else:
#             return [minx, miny, maxx, maxy]
def find_bbox(scoremap, threshold=0.5, scale=4):
    if isinstance(threshold, list):
        bboxes = []
        for i in range(len(threshold)):
            indices = np.where(scoremap > threshold[i])
            try:
                miny, minx = np.min(indices[0] * scale), np.min(indices[1] * scale)
                maxy, maxx = np.max(indices[0] * scale), np.max(indices[1] * scale)
            except:
                bboxes.append([0, 0, 224, 224])
            else:
                bboxes.append([minx, miny, maxx, maxy])
        return bboxes

    else:
        indices = np.where(scoremap > threshold)
        try:
            miny, minx = np.min(indices[0] * scale), np.min(indices[1] * scale)
            maxy, maxx = np.max(indices[0] * scale), np.max(indices[1] * scale)
        except:
            return [0, 0, 224, 224]
        else:
            return [minx, miny, maxx, maxy]

if __name__ == '__main__':
    from PIL import Image
    from cam.utils import *
    input_image_1 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0067_112913.jpg').convert('RGB')
    input_image_2 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0052_112252.jpg').convert('RGB')
    input_image_3 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0007_111029.jpg').convert('RGB')
    input_image_4 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0055_111393.jpg').convert('RGB')
    input_image_5 = Image.open('/data0/caoxz/datasets/CUB_200_2011/images/118.House_Sparrow/House_Sparrow_0130_110985.jpg').convert('RGB')


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
      input_ = seeds.cuda(4, non_blocking=True)
    vgg = models.vgg16(pretrained=True).eval().cuda()

    cam = get_layerCam(input_, vgg)

    for i in range(5):
        mask = cam[i].unsqueeze(0)
        print(mask.shape)
        in_put = input_[i].unsqueeze(0).cpu().detach()
        basic_visualize(in_put.cpu().detach(), mask.type(torch.FloatTensor).cpu(),save_path='./vis/stage_{}.png'.format(i))

