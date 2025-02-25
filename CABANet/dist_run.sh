#!/bin/sh
################### Train #################
CUDA_VISIBLE_DEVICES=0, 1
DATE=`date '+%Y%m%d-%H%M%S'`
echo ${DATE}


# dataset GE HRSC
MODEL="bs-HRSC"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
      train.py --dataset HRSC --end_epoch 100 \
      --lr 0.0001 --train_batchsize 2 --models bs --head fcfpnhead \
      --trans_cnn mit_b2 resnet50 \
      --crop_size 512 512 --use_mixup 0 --use_edge 0 --align 1 --information ${MODEL} > log/${DATE}+${MODEL}.log


#sh cpm.sh


