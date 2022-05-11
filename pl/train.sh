#!/bin/bash
# arch='swin_t'
arch='resnet50'
num_workers=`grep -c processor /proc/cpuinfo`
batch_size=128
echo $num_workers

# python contra_train.py --arch $arch --workers $num_workers
python train_multi.py --arch $arch --workers $num_workers --pretrained true --batch-size $batch_size
# python test.py --arch $arch --workers $num_workers