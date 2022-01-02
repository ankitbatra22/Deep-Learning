import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from os.path import isfile, join
import random
import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

dataset_path = '/Users/ankitbatra/Downloads/brain-tumour-dataset'

train_path = dataset_path + '/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

first_example = train_path + 'BraTS20_Training_001/'

class MRI_Data(Dataset):
    def __init__(self,path):
        self.path = path
        self.patients = [file for file in os.listdir(path) if file not in ['data.csv','README.md']]
        self.masks,self.images = [],[]

        for patient in self.patients:
            for file in os.listdir(os.path.join(self.path,patient)):
                if 'mask' in file.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path,patient,file))
                else: 
                    self.images.append(os.path.join(self.path,patient,file)) 
          
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = io.imread(image)
        image = transform.resize(image,(256,256))
        image = image / 255
        # HWC-> CHW format
        image = image.transpose((2, 0, 1))
        
        
        mask = io.imread(mask)
        mask = transform.resize(mask,(256,256))
        mask = mask / 255
        # HWC-> CHW format
        mask = np.expand_dims(mask,axis=-1).transpose((2, 0, 1))

        #convert the numpy array image and mask to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        #image = torch.from_numpy(image).to_device('cuda')
        #mask = torch.from_numpy(mask).to_device('cuda')
        
        return (image,mask)

dataset = MRI_Data("LGG-dataset/lgg-mri-segmentation/kaggle_3m/")
print("the length of the dataset is: ", dataset.__len__())
print(((dataset.__getitem__(0))))

