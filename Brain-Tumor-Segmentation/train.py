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
from model import UNet
from preprocessing import MRI_Data
import json

with open('config.json') as f:
    config = json.load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the dataset
dataset = MRI_Data(config['dataset_folder'])
print("the length of the dataset is: ", dataset.__len__())

# split the dataset into train and test
train_data, test_data = train_test_split(dataset, test_size=0.15, random_state=42)

# create the dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=config['batch_size'],shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=config['batch_size'])

# create the model
model = UNet(in_channels=3, out_channels=1).to(DEVICE)

criterion = nn.BCEwithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])



