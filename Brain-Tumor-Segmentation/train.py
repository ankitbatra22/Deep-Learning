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
import time

with open('config.json') as f:
    config = json.load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the dataset
dataset = MRI_Data(config['data_folder'])
print("the length of the dataset is: ", dataset.__len__())

# split the dataset into train and test
train_data, test_data = train_test_split(dataset, test_size=0.15, random_state=42)

# create the dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=config['batch_size'],shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=config['batch_size'])

# create the model
model = UNet(in_channels=3, out_channels=1).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])


train_loss = []
val_loss = []

# training loop
for epoch in range(config['num_epochs']):
    print('Epoch: ', epoch)
    start_time = time.time()
    
    running_train_loss = []
    
    for image, mask in train_loader: 
            image = image.to(DEVICE,dtype=torch.float)
            mask = mask.to(DEVICE,dtype=torch.float)
            
            pred_mask = model.forward(image)
            loss = criterion(pred_mask,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())
                              

    else:           
        running_val_loss = []
        
        with torch.no_grad():
            for image,mask in val_loader:
                    image = image.to(device,dtype=torch.float)
                    mask = mask.to(device,dtype=torch.float)                            
                    pred_mask = model.forward(image)
                    loss = criterion(pred_mask,mask)
                    running_val_loss.append(loss.item())
                                    
    
    epoch_train_loss = np.mean(running_train_loss) 
    print('Train loss: {}'.format(epoch_train_loss))                       
    train_loss.append(epoch_train_loss)
    
    epoch_val_loss = np.mean(running_val_loss)
    print('Validation loss: {}'.format(epoch_val_loss))                                
    val_loss.append(epoch_val_loss)
                      
    time_elapsed = time.time() - start_time
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


plt.plot(train_loss,label='train_loss')
plt.plot(val_loss,label='val_loss')
plt.legend()
plt.title('Loss Plot')
plt.show()