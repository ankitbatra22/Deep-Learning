from preprocessing import training_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

"""print(training_data[11])
plt.imshow(training_data[11][0])
plt.show()
"""
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
    self.fc2 = nn.Linear(512, 2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = F.relu(self.conv3(x))
    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.fc1(x))
    x = (self.fc2(x))
    return F.softmax(x, dim=1)

net = Net()
print(net)


optimizer = optim.Adam(net.parameters(), lr = 0.001)
lossFunction = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data])
y = torch.Tensor([i[1] for i in training_data])

percent = 0.1
valSize = int(len(X)*percent)

trainX = X[:-valSize]
trainY = y[:-valSize]
#print(trainX)

print(len(trainX))
print(len(trainY))
#22000

testX = X[-valSize:]
testY = y[-valSize:]

BatchSize = 100
EPOCHS = 1

for epoch in range(EPOCHS):
  print(epoch)
  for i in tqdm(range(0, len(trainX), BatchSize)):
    #print(trainX[i: i + BatchSize].shape)
    batchX = trainX[0:100].view(-1,1,50,50)
    #print(batchX.shape)
    batchY = trainY[0:100]

    #Zero Gradient
    optimizer.zero_grad()
    output = net(batchX)
    loss = lossFunction(output, batchY)
    loss.backward()
    optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
  for i in range(len(testX)):
    pog = torch.argmax(testY[i])
    out = net(testX[i].view(-1,1,50,50))[0]
    predicted = torch.argmax(out)

    if predicted == pog:
      correct+=1
    total+=1

print("Accuracy is: ", round(correct/total, 3))


    
