import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from os.path import isfile, join
import random
import os

dataset_path = '/Users/ankitbatra/Downloads/brain-tumour-dataset'

train_path = dataset_path + '/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

first_example = train_path + 'BraTS20_Training_001/'

print(first_example)

import nibabel as nib
img = nib.load(first_example+'BraTS20_Training_001_flair.nii')
data = img.get_fdata()
print(type(data))


#'/Users/ankitbatra/Downloads/brain-tumour-dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'

DIR = "/Users/ankitbatra/VScode-Workspaces/Grind/ML/Brain-Tumor-Segmentation/LGG-dataset/lgg-mri-segmentation/kaggle_3m"
INPUT_CHANNELS = 3
TARGET_CHANNELS = 1
SIZE = 256
BATCH_SIZE = 32

mri_images_with_tumer = []
mri_images_without_tumer = []
mask_images_with_tumer = []
mask_images_without_tumer = []

patients = os.listdir(DIR)
for patient in (patients):
    if isfile(join(DIR, patient)) == False:
        images = os.listdir(join(DIR, patient))
        mask_images = list(filter(lambda x: x.find('mask') != -1, images))
        mri_images = list(filter(lambda x: x.find('mask') == -1, images))
        
        for mask_image in mask_images:
            mask = np.asarray(load_image(join(DIR, patient, mask_image)))
            if np.amax(mask) != 0:
                mri_images_with_tumer.append(join(patient, mask_image.replace('_mask', '')))
                mask_images_with_tumer.append(join(patient, mask_image))
            else:
                mri_images_without_tumer.append(join(patient, mask_image.replace('_mask', '')))
                mask_images_without_tumer.append(join(patient, mask_image))


class Brain_data(Dataset):
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
        image = image.transpose((2, 0, 1))
        
        
        mask = io.imread(mask)
        mask = transform.resize(mask,(256,256))
        mask = mask / 255
        mask = np.expand_dims(mask,axis=-1).transpose((2, 0, 1))

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return (image,mask)