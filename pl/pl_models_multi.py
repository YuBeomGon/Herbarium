
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision 
import torchvision.transforms as transforms 
import torchvision.models as models

from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import Trainer
# from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchmetrics import Accuracy, F1Score, Specificity

import json
import matplotlib.image as image 
import albumentations as A
import albumentations.pytorch
from sklearn.model_selection import train_test_split
from utils.dataset_multi import *
import custom_models


class HerbClsModel(LightningModule) :
    def __init__(
        self,
        arch: str = 'resnet18',
        pretrained: bool = False,
        lr: float = 0.9,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        num_classes1: int = 15501, # Herbarium num classess(category) 
        num_classes2: int = 2564, # Herbarium num classess(genus) 
        num_classes3: int = 60, # Herbarium num classess(institutions)         
        from_contra : str = './saved_models/contra/',
        steps_per_epoch : int = 100,
        epochs : int = 10,
        # is_contra: bool = False,
    ):
        
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3
        self.from_contra = from_contra
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

        if self.arch in custom_models.__dict__.keys() : 
            # self.model = custom_models.__dict__[self.arch](pretrained=False, img_size=args.img_size)
            self.model = custom_models.__dict__[self.arch](pretrained=self.pretrained)
                    
        shape = self.model.fc.weight.shape
        self.fc1 = nn.Linear(shape[1], self.num_classes1)
        self.fc2 = nn.Linear(shape[1], self.num_classes2)
        self.fc3 = nn.Linear(shape[1], self.num_classes3)
        
        self.criterion = nn.CrossEntropyLoss()    
        print("=> creating model '{}'".format(self.arch))
        self.train_acc1 = Accuracy(top_k=1)
        self.eval_acc1 = Accuracy(top_k=1)
        self.f1 = F1Score(average='macro', num_classes=self.num_classes1)
        self.specificity = Specificity(average='macro', num_classes=self.num_classes1)
        
        self.save_hyperparameters()
        self.r1 = 0.8
        self.r2 = 0.4
        self.r3 = 0.2
        
        torch.nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=.02)
        torch.nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=.02)
        torch.nn.init.trunc_normal_(self.fc3.weight, mean=0.0, std=.02)
        
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x) :
        x = self.model(x)
        o1 = self.fc1(x)
        o2 = self.fc2(x)
        o3 = self.fc3(x)
        return o1, o2, o3

    def training_step(self, batch, batch_idx) :
        images, targets = batch
        tar1, tar2, tar3 = targets
        o1, o2, o3 = self(images)
        
        loss1 = self.criterion(o1, tar1)
        loss2 = self.criterion(o2, tar2)
        loss3 = self.criterion(o3, tar3)
        
        losses = self.r1*loss1 + self.r2*loss2 + self.r3*loss3
        correct=o1.argmax(dim=1).eq(tar1).sum().item()
        total=len(tar1)
        
        #update metric
        self.log('train_loss', losses)
        self.train_acc1(o1, tar1)
        self.log('train_acc', self.train_acc1, prog_bar=True)
        
        #for tensorboard
        logs={"train_loss": losses}
        batch_dictionary={
            'loss':losses,
            'log':logs,
            # info to be used at epoch end
            "correct": correct,
            "total": total
        }        
        
        return batch_dictionary
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
        
        tensorboard_logs = {'losses': avg_loss, "Accuracy": correct/total}
        
        epoch_dictionary={
            # required
            'losses': avg_loss,
            # for logging purposes
            'log': tensorboard_logs
        }

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
    
    def eval_step(self, batch, batch_idx, prefix: str) :
        images, targets = batch
        tar1, tar2, tar3 = targets
        o1, o2, o3 = self(images)
        
        loss1 = self.criterion(o1, tar1)
        loss2 = self.criterion(o2, tar2)
        loss3 = self.criterion(o3, tar3)
        
        losses = self.r1*loss1 + self.r2*loss2 + self.r3*loss3
        
        self.log(f'{prefix}_loss', losses)
        self.eval_acc1(o1, tar1)
        self.log(f'{prefix}_acc1', self.eval_acc1, prog_bar=True)
        self.f1(o1, tar1)
        self.log(f'{prefix}_f1_score', self.f1, prog_bar=True)
        
        if prefix == 'val' :
            correct=o1.argmax(dim=1).eq(tar1).sum().item()
            total=len(tar1) 
            
            #for tensorboard
            logs={"val_loss": losses}
            batch_dictionary={
                'loss':losses,
                'log':logs,
                # info to be used at epoch end
                "correct": correct,
                "total": total
            }  
            
            return batch_dictionary

        return losses           
        
    def validation_step(self, batch, batch_idx) :
        return self.eval_step(batch, batch_idx, 'val')
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
        
        tensorboard_logs = {'loss': avg_loss, "Accuracy": correct/total}
        
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs
        }
        
        # wandb expect None
        # return epoch_dictionary    

    def test_step(self, batch, batch_idx) :
        return self.eval_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self) :
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), 
                              lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch : 0.1 **(epoch //30))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                                        optimizer, 
                                                        epochs              = self.epochs, 
                                                        steps_per_epoch     = self.steps_per_epoch, 
                                                        max_lr              = 0.01, 
                                                        pct_start           = 0.1,  
                                                        div_factor          = 25,   
                                                        final_div_factor    = 1e+4
                                                       ) 
        
        return [optimizer], [scheduler]
    
#     load contra checkpoint in fine tuning
    def load_contra_checkpoint(self, path):
        state_dict = torch.load(path)['state_dict']
        model_state_dict = self.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]

            else:
                print(f"Dropping parameter {k}")

        self.load_state_dict(state_dict)
        
#     model freezing when fine tuning( linear evaluation protocol)
    def model_freeze(self) :
        for param in self.parameters( ) :
            param.requires_grad = False
        
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True
        
class ContraHerbClsModel(HerbClsModel) :  
    def __init__(
        self,
        arch: str = 'resnet18',
        pretrained: bool = False,
        lr: float = 0.9,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        num_classes1: int = 2564, # Herbarium num classess(genus) 
        num_classes2: int = 60, # Herbarium num classess(institutions) 
        # from_contra : str = './saved_models/contra/',
        # is_contra: bool = False,
    ):    
        super().__init__(arch=arch, pretrained=pretrained, lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        
        shape = self.model.fc.weight.shape
        self.model.fc1 = nn.Linear(shape[1], self.num_classes1)
        self.model.fc2 = nn.Linear(shape[1], self.num_classes2)    