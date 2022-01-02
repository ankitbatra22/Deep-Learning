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

train_data, test_data = train_test_split(dataset, test_size=0.15, random_state=42)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=10,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=10)

# visualize
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = (torch.device('cuda') if torch.cuda.is_available()
else torch.device('cpu'))

def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255)
    return image

def mask_convert(mask):
    mask = mask.clone().cpu().detach().numpy()
    return np.squeeze(mask)

def plot_img(no_):
    iter_ = iter(train_loader)
    images,masks = next(iter_)
    images = images.to(device)
    masks = masks.to(device)
    plt.figure(figsize=(20,10))
    for idx in range(0,no_):
         image = image_convert(images[idx])
         plt.subplot(2,no_,idx+1)
         plt.imshow(image)
    for idx in range(0,no_):
         mask = mask_convert(masks[idx])
         plt.subplot(2,no_,idx+no_+1)
         plt.imshow(mask,cmap='gray')
    plt.show()

plot_img(7)
