import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset.cub import CUBDataset
# from utils.dataset.voc import VOCDataset
from utils.dataset.imagenet import  ImageNetDataset
from utils.util import IMAGE_MEAN_VALUE, IMAGE_STD_VALUE


def data_loader(args):
    datasets = ['COCO','VOC2007','VOC2012']

    if args.arch == 'InceptionV3':
        args.resize_size = 229
        args.crop_size = 229
    transform_train = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE)])
    if args.VAL_CROP:
        transform_val = transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size)),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE),
        ])
    else:
        transform_val = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE),
        ])

    if args.dataset == 'CUB':

        img_train = CUBDataset(
            root=args.data_root,
            datalist=os.path.join(args.data_list, 'train.txt'),
            transform=transform_train,
            is_train=True
        )
        img_val = CUBDataset(
            root=args.data_root,
            datalist=os.path.join(args.data_list, 'test.txt'),
            transform=transform_val,
            is_train=False
        )

    elif args.dataset == 'ILSVRC':
        img_train = ImageNetDataset(
            root=os.path.join(args.data_root, 'train'),
            datalist=os.path.join(args.data_list, 'train.txt'),
            transform=transform_train,
            is_train=True
        )
        if args.label_folder:
            val_list = os.path.join(args.data_list, 'val.txt')
        else:
            val_list = os.path.join(args.data_list, 'val.txt')
        img_val = ImageNetDataset(
            root=os.path.join(args.data_root, 'test'),
            datalist=val_list,
            transform=transform_val,
            is_train=False
        )

    else:
        raise Exception("No matching dataset {}.".format(args.dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(img_train)
    else:
        train_sampler = None

    train_loader = DataLoader(img_train,
                              batch_size=args.batch_size,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=args.workers)

    train_loader_2 = DataLoader(img_train,
                              batch_size=args.batch_size,
                              shuffle=False,
                              sampler=None,
                              num_workers=args.workers)

    val_loader = DataLoader(img_val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            sampler=None,
                            num_workers=args.workers)

    return train_loader,train_loader_2, val_loader, train_sampler