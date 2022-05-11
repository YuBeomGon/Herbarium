import os
import json
import numpy as np
import pandas as pd
import math

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision 
import torchvision.transforms as transforms 
from pytorch_lightning import LightningDataModule

import albumentations as A
import albumentations.pytorch
from sklearn.model_selection import train_test_split
import cv2

IMAGE_SIZE=512

train_transforms = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=.8),
        A.VerticalFlip(p=.8),
        A.RandomRotate90(p=.8)]
    ),
    A.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=.8),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
], p=1.0)

val_transforms = A.Compose([
    A.HorizontalFlip(p=.01),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
], p=1.0) 

test_transforms = A.Compose([
    A.HorizontalFlip(p=.01),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
], p=1.0) 

contra_transforms = A.Compose([
    A.RandomScale(scale_limit=.1, p=0.7),
    A.OneOf([
        A.HorizontalFlip(p=.8),
        A.VerticalFlip(p=.8),
        A.RandomRotate90(p=.8)]
    ),
    A.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=.7),
    A.ToGray(p=0.2),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1),
])

class HerbDataset(Dataset) :
    def __init__(self, df, transform=None) :
        self.data_path = '/home/beomgon/pytorch/kaggle/Herbarium/'
        self.df = df
        self.transform = transform
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])               
        print(self.df.shape)
        
    def __len__(self) :
        return len(self.df)
    
    def __getitem__(self, idx) :
        path = self.data_path + self.df.loc[idx, 'image_dir']
        label_cat = self.df.loc[idx, 'category']
        label_gen = self.df.loc[idx, 'genus']
        label_ins = self.df.loc[idx, 'institutions']
        
#         label_cat = torch.tensor(self.df.loc[idx, 'category'])
#         label_gen = torch.tensor(self.df.loc[idx, 'genus'])
#         label_ins = torch.tensor(self.df.loc[idx, 'institutions'])
        
#         labels = torch.stack([label_cat, label_gen, label_ins], dim=0)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform : 
            timage = self.transform(image=image)
            image = timage['image']
            
        image = self.get_tensor(image)
        
        return image, (label_cat, label_gen, label_ins)
        
        # return image, labels
        
    def get_tensor(self, image) :
        image = image/255.
        image = (image - self.image_mean[None, None, :]) / self.image_std[None, None,:]
        image =  torch.tensor(image, dtype=torch.float32)
        image = image.permute(2,0,1)
        
        return image

class ContraHerbDataset(HerbDataset) :
    def __init__(self, df, transform=None) :
        super(ContraHerbDataset, self).__init__(df, transform=transform)
    
    def __getitem__(self, idx) :
        path = self.df.loc[idx, 'image_dir']
        label = self.df.loc[idx, 'category']
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform : 
            timage1 = self.transform(image=image)
            timage2 = self.transform(image=image)
            image1 = timage1['image']
            image2 = timage2['image']
            
            image1 = self.get_tensor(image1)
            image2 = self.get_tensor(image2)
        else :
            image2 = image1 = get_tensor(image)
        
        return image1, image2, label
    
class HerbDataModule(LightningDataModule):
    def __init__(self, df, batch_size=4, workers=4, transform=None):
        super().__init__()
        self.df = df
        print(self.df.shape)
        self.transform = transform
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])    
        self.batch_size = batch_size
        self.seed = 0
        self.targets = self.df['category']
        self.ratio = 0.25
        self.shuffle = True
        self.workers = workers
        
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        
    def prepare_data(self):
        pass 

    def setup(self, stage=None):
        self.train_df, self.test_df = train_test_split(self.df, 
                                                       # np.arange(len(self.df)), # when using index
                                                       test_size=self.ratio,
                                                       shuffle=self.shuffle,
                                                       stratify=self.targets,
                                                       random_state=self.seed,
                                                      )
        
        self.train_df.reset_index(inplace=True, drop=True)
        self.test_df.reset_index(inplace=True, drop=True)
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = HerbDataset(self.train_df, transform=self.train_transforms)
        self.test_dataset = HerbDataset(self.test_df, transform=self.test_transforms)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.workers
        )       
    
class ContraHerbDataModule(HerbDataModule):
    def __init__(self, df, batch_size, transform=None):
        super().__init__(df, batch_size, transform=transform)
        self.contra_transforms = contra_transforms

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = ContraHerbDataset(self.train_df, transform=self.contra_transforms)
        self.test_dataset = ContraHerbDataset(self.test_df, transform=self.contra_transforms)
      