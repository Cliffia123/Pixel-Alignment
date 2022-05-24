# Pixel-Alignment
Exploring Pixel Alignment on Shallow Feature for Weakly Supervised Object Localization


## Data Preparation
- [CUB-200-2011 download link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

## Code Structure
```
├── datalist
│   ├── CUB
│   │   ├── bounding_boxes.txt
│   │   ├── image_class_labels.txt
│   │   ├── images.txt
│   │   ├── sizes.txt
│   │   ├── test.txt
│   │   └── train.txt
│   └── ILSVRC
│       ├── gt_ImageNet.pickle
│       ├── image_test.py
│       ├── sizes.txt
│       ├── train.txt
│       ├── val_folder.txt
│       └── val.txt
├── install.sh
├── main2.py
├── main.py
├── main_we.py
├── models
│   └── model.py
├── network
│   ├── core
│   │   ├── constants.py
│   │   ├── entropy.py
│   │   ├── heads.py
│   │   ├── modules.py
│   │   └── selflearning.py
│   ├── erasing.py
│   ├── evaluator_change.py
│   ├── evaluator.py
│   ├── __init__.py
│   ├── loss
│   │   ├── entropy.py
│   │   └── selflearning.py
│   ├── loss.py
│   ├── main_model.py
│   ├── main_model_we.py
│   ├── model.py
│   ├── net_factory.py
│   ├── resnet.py
│   ├── unet.py
│   ├── vgg16_acol_ori.py
│   ├── vgg16_acol.py
│   └── vgg16_merge.py
├── script
│   ├── evaluate_cub.sh
│   ├── evaluate_imagenet.sh
│   ├── train_cub.sh
│   └── train_imagenet.sh
├── setup.py
├── test.py
├── train_log
│   └── cxz
│       ├── cub_coarse_best_model
│       │   ├── last_epoch_cub.pth
│       │   └── last_epoch_ilsvrc.pth
│       ├── evaluator
│       │   ├── classifier.pth.tar
│       │   ├── evaluator_ilsvrc.pth.tar
│       │   └── evaluator.pth.tar
│       ├── model_best_eil_cub.pth.tar
└── utils
    ├── cam
    │   ├── basecam.py
    │   ├── layercam.py
    │   └── utils
    │       ├── imagenet.py
    │       ├── __init__.py
    │       └── resources
    │           ├── imagenet_class_index.json
    │           └── __init__.py
    ├── dataset
    │   ├── cub.py
    │   └── imagenet.py
    ├── evaluator.py
    ├── util_args.py
    ├── util_cam.py
    ├── util_loader.py
    └── util.py
```
## Execution
##### Download

```
git clone https://github.com/Cliffia123/IJCNN
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz
```

##### Evaluate
```
1. For CUB database
./script/evaluate_cub.sh
```
##### 
```
#!/bin/bash
# VGG16 script
gpu=0
arch=vgg16_acol
name=cxz
dataset=CUB   
data_root="/data0/caoxz/datasets/CUB_200_2011/images"  CUB数据集存放的位置
epoch=1
decay=40
batch=32
wd=1e-4
lr=0.001
bbox_mode='classical'

CUDA_VISIBLE_DEVICES=${gpu} python main_we.py \    测试的执行文件
--multiprocessing-distributed \
--world-size 1 \
--workers 32 \
--arch ${arch} \
--name ${name} \
--dataset ${dataset} \
--data-root ${data_root} \
--pretrained True \
--batch-size ${batch} \
--epochs ${epoch} \
--lr ${lr} \
--LR-decay ${decay} \
--wd ${wd} \
--nest True \
--erase-thr 0.6 \
--acol-cls False \
--VAL-CROP True \
--evaluate True \
--cam-thr 0.5 \   #测试的阈值，可以调整0-1之间，哪个效果最好选哪个
--sim_fg_thres 0.6 \
--sim_bg_thres 0.1 \ 
--resume train_log/cxz/model_best_eil_cub.pth.tar \    测试的模型
--cls_checkpoint train_log/cxz/evaluator/evaluator.pth.tar

```
