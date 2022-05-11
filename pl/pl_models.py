
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from torchmetrics import Accuracy, F1Score, Specificity

import json
import matplotlib.image as image 
import albumentations as A
import albumentations.pytorch
from sklearn.model_selection import train_test_split
from utils.dataset import *


class HerbClsModel(LightningModule) :
    def __init__(
        self,
        arch: str = 'resnet18',
        pretrained: bool = False,
        lr: float = 0.9,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        num_classes: int = 15501, # Herbarium num classess(category) 
        from_contra : str = './saved_models/contra/',
        # is_contra: bool = False,
    ):
        
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.from_contra = from_contra
        
        if self.arch not in models.__dict__.keys() : 
            # self.model = custom_models.__dict__[self.arch](pretrained=False, img_size=args.img_size)
            self.model = custom_models.__dict__[self.arch](pretrained=False)
        else :
            print('only resnet is supported') 
            self.model = models.__dict__[self.arch](pretrained=self.pretrained) 
        
        shape = self.model.fc.weight.shape
        self.model.fc = nn.Linear(shape[1], self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
            
        print("=> creating model '{}'".format(self.arch))
        self.train_acc1 = Accuracy(top_k=1)
        self.eval_acc1 = Accuracy(top_k=1)
        self.f1 = F1Score(average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(average='macro', num_classes=self.num_classes)
        
        self.save_hyperparameters()
        
        torch.nn.init.trunc_normal_(self.model.fc, mean=0.0, std=.02)
        
    def forward(self, x) :
        return self.model(x)

    def training_step(self, batch, batch_idx) :
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        correct=outputs.argmax(dim=1).eq(targets).sum().item()
        total=len(targets)
        
        #update metric
        self.log('train_loss', loss)
        self.train_acc1(outputs, targets)
        self.log('train_acc', self.train_acc1, prog_bar=True)
        
        #for tensorboard
        logs={"train_loss": loss}
        batch_dictionary={
            'loss':loss,
            'log':logs,
            # info to be used at epoch end
            "correct": correct,
            "total": total
        }        
        
        return batch_dictionary
    
    def training_epoch_end(self, outputs):
        # if(self.current_epoch==1):
        #     sampleImg=torch.rand((1,3,IMAGE_SIZE,IMAGE_SIZE))
        #     self.logger.experiment.add_graph(self.model(), sampleImg)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        # logging histograms
        # self.custom_histogram_adder()
        
        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
        
        # creating log dictionary
        # self.logger.experiment.add_scalar('loss/train', avg_loss, self.current_epoch)
        # self.logger.experiment.add_scalar('acc/train', correct/total, self.current_epoch)
        tensorboard_logs = {'loss': avg_loss, "Accuracy": correct/total}
        
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs
        }
 
        # wandb expect None
        # return epoch_dictionary  

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
    
    def eval_step(self, batch, batch_idx, prefix: str) :
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        self.log(f'{prefix}_loss', loss)
        self.eval_acc1(outputs, targets)
        self.log(f'{prefix}_acc1', self.eval_acc1, prog_bar=True)
        self.f1(outputs, targets)
        self.log(f'{prefix}_f1_score', self.f1, prog_bar=True)
        self.specificity(outputs, targets)
        self.log(f'{prefix}_specificity', self.specificity, prog_bar=True)    
        
        if prefix == 'val' :
            correct=outputs.argmax(dim=1).eq(targets).sum().item()
            total=len(targets) 
            
            #for tensorboard
            logs={"val_loss": loss}
            batch_dictionary={
                'loss':loss,
                'log':logs,
                # info to be used at epoch end
                "correct": correct,
                "total": total
            }  
            
            return batch_dictionary

        return loss            
        
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
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch : 0.1 **(epoch //30))
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