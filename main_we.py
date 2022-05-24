import os
import random
import numpy as np
from numpy import *
import warnings
from tqdm import tqdm
# from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# from torchvision.utils import make_grid
import torch.nn.functional as F

import network as models
from utils.util_args import get_args
from utils.util_cam import *
from utils.util_loader import data_loader
from utils.util import \
    accuracy, adjust_learning_rate, \
    save_checkpoint, load_model, AverageMeter, IMAGE_MEAN_VALUE, IMAGE_STD_VALUE, calculate_IOU, draw_bbox, save_images, \
    save_erased_images
import cv2
# from network.vgg16_acol import VGG
from network.main_model_we import FineModel
from network.main_model import CoarseModel
from network.evaluator import Evaluator
from network.loss import *
from network.core.selflearning import *
from utils.util import *
# from network.gcn_network.net_factory import get_network_fn
torch.set_num_threads(4)
best_acc1 = 0
best_loc1 = 0

def main():
    args = get_args()
    print(args)
    # 可视化训练日志存储
    args.log_folder = os.path.join('train_log/', args.name)

    # ddt存放路径
    args.vis_path = '/data0/caoxz/BB/Boundary-en/vis'

    # ddt训练图片存储路径
    args.erased_folder = '/data0/caoxz/BB/Boundary-en'

    if args.arch == 'vgg':
        args.top_k = 80
    if args.arch == 'inception':
        args.threshold = 0.1

    if not os.path.join(args.log_folder):
        os.makedirs(args.log_folder)

    # 配置：seed, gpu, dist_url, distribute,
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_loc1
    args.gpu = gpu
    global writer
    # 检测gpu等配置
    # if args.gpu == 0 and not args.evaluate:
    # writer = SummaryWriter(logdir=args.log_folder)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.dataset == "CUB":
        num_classes = 200
    elif args.dataset == "ILSVRC":
        num_classes = 1000
    else:
        raise Exception("No dataset named {}".format(args.dataset))

    # Define Model 模型架构，基础是vgg16

    # 原始
    # model = vgg16_acol._vgg(
    #     pretrained=True,
    #     progress=True,
    #     num_classes=num_classes
    # )
    # model = VGG(num_classes=num_classes)
    # args.training_cls = False
    # floder = os.path.join(args.log_folder, 'cub_coarse_best_model')+'/last_epoch.pth'

    #train evaluator

    model = FineModel(args.resume, cls_checkpoint=args.cls_checkpoint, args=args, num_classes=num_classes).cuda()
    args.training_cls = False
    # model = CoarseModel(num_classes = num_classes).cuda()
    # args.training_cls = True
    param_features = []
    param_classifier = []

    # Give different learning rate
    for name, parameter in model.named_parameters():
        if 'fc.' in name or '.classifier' in name:
            param_classifier.append(parameter)
        else:
            param_features.append(parameter)
    optimizer = torch.optim.SGD([
        {'params': param_features, 'lr': args.lr},
        {'params': param_classifier, 'lr': args.lr * args.lr_ratio}],
        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nest)


    # Change the last fully connected layers.
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint 从最新的检查点开始恢复
    # if args.resume:
    #     model, optimizer = load_model(model, optimizer, args)

    # Loading training, validation, ddt dataset
    cudnn.benchmark = True
    train_loader, train_loader_2, val_loader, train_sampler = data_loader(args)


    criterion = [torch.nn.CrossEntropyLoss().cuda(), AreaLoss(topk=25).cuda(),
                 DenseCRFLoss(args.crf_lambda, args.crf_sigma_rgb, args.crf_sigma_xy, args.crf_scale_factor).cuda(), WeightedEntropyLoss().cuda(),
                MaxSizePositiveFcams().cuda(), SelfLearningFcams().cuda()]


    if args.evaluate:
        # model, optimizer = load_model(model, optimizer, args)
        val_acc1, val_acc5, top1_loc, top5_loc, gt_loc, val_loss = evaluate_loc(val_loader, model, 0, criterion,
                                                                                args)
        return

    print("============> 1. 开始初始化训练 <============")
    best_acc1 = 0
    best_loc1 = 0
    best_acc1 = train_mode(model, train_sampler, optimizer, train_loader, val_loader, ngpus_per_node ,best_acc1, best_loc1, criterion,args)

def train_mode(model, train_sampler, optimizer, train_loader, val_loader, ngpus_per_node, best_acc1, best_loc1, criterion, args):
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        val_acc1, train_loss = train(train_loader, model, optimizer, epoch, criterion, args)

        # 默认loc为False
        # val_acc1, core = evaluate(val_loader, model, criterion, epoch, args)

        val_acc1, val_acc5, top1_loc, top5_loc, gt_loc, val_loss = evaluate_loc(val_loader, model, epoch, criterion, args)

        if args.training_cls==False:
            is_best = top1_loc > best_loc1
            best_loc1 = max(top1_loc,best_loc1)

        if args.training_cls:
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)

        if args.gpu == 0:
            if args.training_cls:
                print("Until %d epochs, Best Acc@1 %.3f" % (epoch + 1, best_acc1))
            else:
                print("Until %d epochs, Best Acc@1 %.3f" % (epoch + 1, best_loc1))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': val_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.log_folder)
    return best_acc1

# 训练流程开始
def train(train_loader, model, optimizer, epoch, criterion, args):
    # AverageMeter for Performance  平均性能表
    losses = AverageMeter('Loss')
    loss_Eval = AverageMeter('ClsLoss', ':.4e')

    loss_AC = AverageMeter('ACLoss', ':.4e')
    loss_Unet = AverageMeter('UnetLoss', ':.4e')
    loss_Entropy = AverageMeter('BgLoss', ':.4e')

    loss_CRF = AverageMeter('CRFLoss', ':.4e')
    loss_KL = AverageMeter('KLLoss', ':.4e')
    loss_Size = AverageMeter('SizeLoss', ':.4e')
    loss_Cam = AverageMeter('CamLoss', ':.4e')


    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    sl_mask_builder = GetFastSeederSLFCAMS()

    model.train()
    train_t = tqdm(train_loader)  # 进度条██████████
    for i, (images, mage_id, image_class, image_name, target) in enumerate(train_t):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)
        #
        # target_layer = model.module.vgg.model.features[30]
        # target_layer.register_forward_hook(model.module.forward_hook)
        # target_layer.register_backward_hook(model.module.backward_hook)
        # images = images.requires_grad_()

        output = model(images)
        #
        # predicted_class = output['score'].max(1)[-1]
        # one_hot_output = torch.FloatTensor(output['score'].size(0), output['score'].size()[-1]).zero_()
        # for i in range(output['score'].size(0)):
        #     one_hot_output[i][predicted_class] = 1
        # one_hot_output = one_hot_output.cuda(args.gpu, non_blocking=True)
        # # model.zero_grad()
        # output['score'].backward(gradient=one_hot_output, retain_graph=True)
        # gradients = model.module.gradients['value'].clone().detach()
        # activation = model.module.activations['value'].clone().detach()
        #
        # cam , weight = get_nor_layercam(activation, gradients)


        # for i in range(gradients.size(0)):
        #     mask_ = cam[i].unsqueeze(0)
        #     in_put = images[i].unsqueeze(0).cpu().detach()
        #     basic_visualize(in_put.cpu().detach(), mask_.type(torch.FloatTensor).cpu(),
        #                     save_path='./re/s_{}.png'.format(i))
        #
        # for i in range(gradients.size(0)):
        #     mask_ = weight[i].unsqueeze(0)
        #     in_put = images[i].unsqueeze(0).cpu().detach()
        #     basic_visualize(in_put.cpu().detach(), mask_.type(torch.FloatTensor).cpu(),
        #                     save_path='./re/st_{}.png'.format(i))

        loss_eval = criterion[0](output['score'], target)  # for evaluation
        # loss_unet = criterion[0](output['score_unet'], target)  # for evaluation

        acc1, acc5 = accuracy(output['score'], target, topk=(1, 5))

        if args.training_cls:
            loss = loss_eval
        else:
            loss_ac = criterion[1](output['unet_activate'], output['score'], output['activate_features'])
            # loss_crf = criterion[2](images, output['unet_activate'])
            loss_entropy = criterion[3](output['unet_activate'])

            # loss_entropy_cam = criterion[3](output['cam'])

            # loss_kl = criterion[4](output['score_bg'])
            # loss_size = criterion[4](output['unet_activate'])
            # seeds = torch.zeros((images.size(0),224, 224), requires_grad=False, dtype=torch.long)
            # for i in range(images.size(0)):
            #     pulled_cam = output['cam'][i].unsqueeze(1)
            #     pulled_cam = normalize_minmax(pulled_cam)
            #     seeds[i] = sl_mask_builder(pulled_cam).squeeze(1)
            # from torch.autograd import Variable
            # images = Variable(images, requires_grad=True)  # 生成变量
            # for param in model.parameters():
            #     print('{}:grad->{}'.format(param, param.grad.shape))
            # loss_cam = criterion[5](output['unet_activate'], cam.squeeze(1), weight.squeeze(1))
            loss = loss_eval + 0.5*loss_ac + loss_entropy

            loss_AC.update(loss_ac.data.item(), images.size(0))
            loss_Entropy.update(loss_entropy.data.item(), images.size(0))
            # loss_Cam.update(loss_cam.data.item(), images.size(0))
            # loss_CRF.update(loss_crf.data.item(), images.size(0))
            # loss_Size.update(loss_size.data.item(), images.size(0))
            # loss_KL.update(loss_kl.data.item(), images.size(0))

        loss_Eval.update(loss_eval.data.item(), images.size(0))
        # loss_Unet.update(loss_unet.data.item(), images.size(0))

        losses.update(loss.data.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # if args.training_cls:
        description = "[T:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}". \
                format(epoch, args.epochs, top1.avg, top5.avg, losses.avg)

        train_t.set_description(desc=description)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg, losses.avg

def evaluate_loc(val_loader, model, epoch, criterion, args):
    losses = AverageMeter('Loss')
    loss_Eval = AverageMeter('ClsLoss', ':.4e')

    loss_AC = AverageMeter('ACLoss', ':.4e')
    loss_Unet = AverageMeter('UnetLoss', ':.4e')
    loss_Entropy = AverageMeter('BgLoss', ':.4e')
    loss_Size = AverageMeter('SizeLoss', ':.4e')
    loss_Cam = AverageMeter('CamLoss', ':.4e')


    loss_CRF = AverageMeter('CRFLoss', ':.4e')
    loss_KL = AverageMeter('KLLoss', ':.4e')


    top1_cls = AverageMeter('Acc@1')
    top5_cls = AverageMeter('Acc@5')

    # image
    gt_bbox = load_bbox(args)
    cnt = 0
    cnt_false_top1 = 0
    cnt_false_top5 = 0
    hit_known = 0
    hit_top1 = 0
    hit_top5 = 0
    iou_list = []
    image_mean = torch.reshape(torch.tensor(IMAGE_MEAN_VALUE), (1, 3, 1, 1))
    image_std = torch.reshape(torch.tensor(IMAGE_STD_VALUE), (1, 3, 1, 1))
    # sl_mask_builder = GetFastSeederSLFCAMS()

    model.eval()
    with torch.no_grad():
        val_t = tqdm(val_loader)
        for i, (images, image_ids, image_class, image_name, target) in enumerate(val_t):
            target = target.cuda(args.gpu, non_blocking=True)


            output = model(images)

            loss_eval = criterion[0](output['score'], target)  # for evaluation
            acc1, acc5 = accuracy(output['score'], target, topk=(1, 5))

            if args.training_cls:
                loss = loss_eval
            else:
                # loss_ac = criterion[1](output['unet_activate'], output['score'], output['activate_features'])
                # loss_crf = criterion[2](images, output['unet_activate'])
                # loss_entropy = criterion[3](output['unet_activate'])
                # loss_kl = criterion[4](output['score_bg'])
                # loss_size = criterion[5](output['unet_activate'])
                # seeds = torch.zeros((images.size(0), 224, 224), requires_grad=False, dtype=torch.long)
                # for i in range(images.size(0)):
                #     pulled_cam = output['cam'][i].unsqueeze(1)
                #     pulled_cam = normalize_minmax(pulled_cam)
                #     seeds[i] = sl_mask_builder(pulled_cam).squeeze(1)

                loss = loss_eval
                # loss_AC.update(loss_ac.data.item(), images.size(0))
                # loss_Entropy.update(loss_entropy.data.item(), images.size(0))
                # loss_CRF.update(loss_crf.data.item(), images.size(0))
                # loss_Size.update(loss_size.data.item(), images.size(0))
                # loss_KL.update(loss_kl.data.item(), images.size(0))

            loss_Eval.update(loss_eval.data.item(), images.size(0))
            # loss_Unet.update(loss_unet.data.item(), images.size(0))

            losses.update(loss.data.item(), images.size(0))
            top1_cls.update(acc1[0], images.size(0))
            top5_cls.update(acc5[0], images.size(0))

            description = "[T:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}". \
                format(epoch, args.epochs, top1_cls.avg, top5_cls.avg, losses.avg)
            val_t.set_description(desc=description)

            _, pred = output['score'].topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_1 = correct[:1].flatten(1).float().sum(dim=0)
            correct_5 = correct[:5].flatten(1).float().sum(dim=0)

            # Get cam 得到得到分类器A和B中特征最突出的热力图像
            #选取cams图片

            cam_ddt_list = get_cam(model=model, args=args)
            # cam_ddt_list = attmap
            image_ = images.clone().detach().cpu() * image_mean + image_std
            image_ = image_.numpy().transpose(0, 2, 3, 1)

            image_ = image_[:, :, :, ::-1] * 255

            # mask = cam_ddt_list.new_ones((images.size(0), 1, 224, 224))

            # pos = torch.le(cam_ddt_list, 0.94)
            # mask[pos.data] = 0.
            # cam_ddt_list = cam_ddt_list * mask

            cam = cam_ddt_list.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)
            # visualization(images, cam, image_name, image_ids, gt_bbox)

            for j in range(images.size(0)):
                if args.dataset == "CUB":
                    box = gt_bbox[int(image_ids[j])]
                else:
                    box = gt_bbox[image_name[j][:-5]]
                estimated_bbox, blend_bbox, blend_0 = generate_bbox(image_[j],cam[j],box[0],args.cam_thr, args)

                iou = calculate_IOU(box, estimated_bbox)


                saving_folder = os.path.join(args.log_folder, 'result_total')
                if not os.path.isdir(saving_folder):
                    os.makedirs(saving_folder)
                cv2.imwrite(os.path.join(saving_folder, image_name[j]), blend_bbox)

                cnt += 1
                if iou >= 0.5:
                    iou_list.append(iou)
                    hit_known += 1
                    if correct_5[j] > 0:
                        hit_top5 += 1
                        if correct_1[j] > 0:
                            hit_top1 += 1
                        elif correct_1[j] == 0:
                            cnt_false_top1 += 1
                    elif correct_5[j] == 0:
                        cnt_false_top1 += 1
                        cnt_false_top5 += 1
                else:
                    if correct_5[j] > 0:
                        if correct_1[j] == 0:
                            cnt_false_top1 += 1
                    elif correct_5[j] == 0:
                        cnt_false_top1 += 1
                        cnt_false_top5 += 1

            # save_images('results_total', 0, i, blend_tensor, args)

            description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
                format(epoch, args.epochs, top1_cls.avg, top5_cls.avg, losses.avg)
            val_t.set_description(desc=description)

        loc_gt = hit_known / cnt * 100
        loc_top1 = hit_top1 / cnt * 100
        loc_top5 = hit_top5 / cnt * 100
        cls_top1 = (1 - cnt_false_top1 / cnt) * 100
        cls_top5 = (1 - cnt_false_top5 / cnt) * 100

        if args.gpu == 0:
            print("Evaluation Result:\n"
                  "LOC GT:{0:6.2f} Top1: {1:6.2f} Top5: {2:6.2f}\n"
                  "CLS TOP1: {3:6.3f} Top5: {4:6.3f}".
                  format(loc_gt, loc_top1, loc_top5, cls_top1, cls_top5))
    return cls_top1, cls_top5, loc_top1, loc_top5, loc_gt, losses.avg

if __name__ == '__main__':
    main()
