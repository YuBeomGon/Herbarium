import argparse
import os
from typing import Optional
import pandas as pd
import logging
from datetime import datetime
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchmetrics import Accuracy, F1Score, Specificity

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import Trainer
# from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from utils.dataset_multi import HerbDataset
from utils.losses import SupConLoss, FocalLoss
from utils.dataset_utils import LabelEncoder
import custom_models
from pl_models_multi import *

# import custom_models

parser = argparse.ArgumentParser(description='PyTorch Lightning ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default='/home/beomgon/pytorch/kaggle/Herbarium/dataset/',
                    help='path to dataset (default: ./lbp_data/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=12, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--accelerator', '--accelerator', default='gpu', type=str, help='default: gpu')

parser.add_argument('--devices', default=4, type=int, help='number of gpus, default 2')
parser.add_argument('--img_size', default=400, type=int, help='input image resolution in swin models')
parser.add_argument('--num_classes', default=15501, type=int, help='number of herbarium classes, category')
parser.add_argument('--saved_dir', default='./saved_models/tunning', type=str, help='directory for model checkpoint')
parser.add_argument('--from_contra', default='./saved_models/contra', type=str, help='directory for model checkpoint')

parser.add_argument('--pretrained', default=False, type=bool, help='use pretrained model or not')


if __name__ == "__main__":
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    args = parser.parse_args()
    print('workers', args.workers)
    if torch.cuda.is_available() :
        args.accelerator = 'gpu'
        args.devices = torch.cuda.device_count()
        
    args.img_size = IMAGE_SIZE
    logger_tb = TensorBoardLogger('./tuning_logs' +'/' + args.arch, name=now)
    logger_wandb = WandbLogger(project='herb_clf', name=now, mode='online') # online or disabled    
    
    trainer_defaults = dict(
        callbacks = [
            # the PyTorch example refreshes every 10 batches
            TQDMProgressBar(refresh_rate=10),
            # save when the validation top1 accuracy improves
            ModelCheckpoint(monitor="val_acc1", mode="max",
                            dirpath=args.saved_dir + '/' + args.arch,
                            filename='herb_tunning_{epoch}_{val_acc1:.2f}'),  
            ModelCheckpoint(monitor="val_acc1", mode="max",
                            dirpath=args.saved_dir + '/' + args.arch,
                            filename='herb_tunning_best'),  
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],    
        # plugins = "deepspeed_stage_2_offload",
        # plugins = "deepspeed_stage_3",
        precision = 16,
        max_epochs = args.epochs,
        accelerator = args.accelerator, # auto, or select device, "gpu"
        devices = -1, # use of available gpus
        logger = [logger_tb, logger_wandb],
        benchmark = True,
        # strategy = "ddp",
        strategy = "ddp_sharded",
        
        )
    
    # csv file correction for training
    df = pd.read_csv(args.data_path + 'df.csv')
    le = LabelEncoder(df)
    df = le.get_integer_labels()
    # df.image_dir = df.image_dir.apply(lambda x : re.sub('../dataset','./dataset', x))
    
    # set dataModule
    dm = HerbDataModule(df, batch_size=args.batch_size, workers=args.workers, transform=None) 
    
    dm.setup(stage='fit')
    steps_per_epoch = len(dm.train_dataloader())
    
    model = HerbClsModel(
        arch=args.arch,
        pretrained=args.pretrained,
        lr = args.lr,
        weight_decay=args.weight_decay,
        from_contra=args.from_contra,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs)    
    
    path = args.from_contra + '/' + args.arch
    if os.path.isdir(path) and 'herb-contra_best.ckpt' in os.listdir(path) :
        print('checkpoint is loaded from ', path)
        model.load_contra_checkpoint(path + '/herb-contra_best.ckpt')   
#         model freeze except last fcn layer
        print('model freeze except last fc layer')
        model.model_freeze()
     
    trainer = Trainer(**trainer_defaults)
    trainer.fit(model, dm)  
    trainer.test(model, dm)