#!/bin/bash
gpu=6
arch=vgg16_acol
name=cxz
dataset=CUB
data_root=
3"/data0/caoxz/datasets/CUB_200_2011/images"
epoch=1
decay=40
batch=32
wd=1e-4
lr=0.001
bbox_mode='classical'

CUDA_VISIBLE_DEVICES=${gpu} python main_we.py \
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
--cam-thr 0.7 \
--sim_fg_thres 0.6 \
--sim_bg_thres 0.1 \
--resume train_log/cxz/model_best_eil_cub.pth.tar \
--cls_checkpoint train_log/cxz/evaluator/evaluator.pth.tar
