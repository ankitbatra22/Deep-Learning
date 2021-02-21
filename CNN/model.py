from preprocessing import training_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


KERNAL = 5

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    # Layer 1 sees a 1x50x50 image tensors
    self.conv1 = nn.Conv2d(1, 32, KERNAL)
    # Layer 2 sees a 32x46x46
    self.conv2 = nn.Conv2d(32, 64, KERNAL)
    # Layer 3 sees a 64x42x42
    self.conv3 = nn.Conv2d(64, 128, KERNAL)
    #output of third conv layer is 128x38x38
    # Max pooling
    self.pool1 = nn.MaxPool2d((2,2))
    self.pool2 = nn.MaxPool2d((2,2))
    #self.pool3 = nn.MaxPool2d((2,2))
    self.fc1 = nn.Linear(128*5*5, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 1)
    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = F.relu(self.conv3(x))
    x = torch.flatten(x, start_dim=1)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = (self.fc3(x))
   # print(x)
    x = F.sigmoid(x)
   # print(x)
    return x

net = Net()
print(net)


optimizer = optim.Adam(net.parameters(), lr = 0.001)
lossFunction = nn.BCELoss()

X = torch.Tensor([i[0] for i in training_data])
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

percent = 0.9
valSize = int(len(X)*percent)

trainX = X[:valSize]
#print(trianX.shape)
trainY = y[:valSize]
#print(trainY.shape)
#22000

testX = X[valSize:]
testY = y[valSize:]

BatchSize = 100
EPOCHS = 8



for epoch in range(EPOCHS):
  print(epoch)
  for i in tqdm(range(0, len(trainX), BatchSize)):
    batchX = trainX[i:i+BatchSize].view(-1,1,50,50)
    #print(batchX)
    #print(batchX.shape)
    batchY = trainY[i:i+BatchSize].view(-1,1)
    #print(trainY[i:i+BatchSize].shape)
    #batchX = trainX[1].view(-1,1,50,50)
    #batchY = trainY[1].view(-1,1)

    #Zero Gradient
    optimizer.zero_grad()
    output = net(batchX)
    #print(output)
    loss = lossFunction(output, batchY)
    loss.backward()
    optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
  for i in range(len(testX)):
    pog = (testY[i])
    #print(pog)
    out = net(testX[i].view(-1,1,50,50))
    #print(out.item())
    predicted = round(out.item(), 0)
    if int(predicted) == pog:
      correct+=1
    total+=1

print("Accuracy is: ", round(correct/total, 3))


    

## TO DO:
# Change to color
# Deeper Network (dropout, earlystop)
# Print graphs for loss decrease

