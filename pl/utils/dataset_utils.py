
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset, DataLoader
from sklearn import preprocessing


# change labels to integer
class LabelEncoder() :
    def __init__(self, df) :
        self.df = df
        self.le_cate = preprocessing.LabelEncoder()
        self.le_gen = preprocessing.LabelEncoder()
        self.le_inst = preprocessing.LabelEncoder()

    def fit(self) :
        self.le_cate.fit(self.df.category)
        self.le_gen.fit(self.df.genus)
        self.le_inst.fit(self.df.institutions)
        
    def transform(self) :
        self.df.category = self.le_cate.transform(self.df.category)
        self.df.genus = self.le_gen.transform(self.df.genus)
        self.df.institutions = self.le_inst.transform(self.df.institutions)
        
    def get_integer_labels(self) :
        self.fit()
        self.transform()
        return self.df
        
    def inverse_transform(self, preds, s) :
        if s == 'category' :
            return self.le_cate.inverse_transform(preds)
        if s == 'genus' :
            return self.le_gen.inverse_transform(preds)
        if s == 'institutions' :
            return self.le_inst.inverse_transform(preds)            
        
def get_loader_train_test() :
    pass
        
        


    