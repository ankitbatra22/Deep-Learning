import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
### MAX POOLING VS GLOBAL POOLING?, MINI BATCH, WHAT POOLING LAYER DOES?, why conv layers first and what final output of the conv layer means?
# shape is (batchSize, channels, width, height)
x = torch.randn(1,1,50,50)
print(x.shape)

#conv1 = torch.nn.Conv2d(32, 64, 3, padding=1, stride=2)
conv2 = torch.nn.Conv2d(1,32,5)
#print(conv1.stride)
#print(conv1(x).shape)

pool = torch.nn.MaxPool2d(2,2)
#print(pool(conv1(x)).shape)

#print(conv2(x).shape)

y = torch.randn(1,32,46,46)
#print(y)
conv3 = torch.nn.Conv2d(32, 64, 5)
print(conv3(y).shape)"""


"""conv1 = nn.Conv2d(1, 32, 5)
    # Layer 2 sees a 32x46x46
conv2 = nn.Conv2d(32, 64, 5)
    # Layer 3 sees a 64x42x42
conv3 = nn.Conv2d(64, 128, 5)
    #output of third conv layer is 128x38x38
    # Max pooling
pool1 = nn.MaxPool2d((2,2))
pool2 = nn.MaxPool2d((2,2))

x = torch.randn(1,1,50,50)

x = (conv3(pool2(conv2(pool1(conv1(x))))))
x = torch.flatten(x, start_dim=1)
print(x.shape)"""


x = torch.randn(10,2)
#print(x)

print(F.softmax(x, dim=1))

