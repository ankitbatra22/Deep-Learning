import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#define two datasets (ALWAYS)
train = datasets.MNIST("", train=True, download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))
                      
test = datasets.MNIST("", train=False, download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))

# Batch size is how many at a time you want to pass to the model. We want the dataset to generalize. The changes (optimizations) that are adjusted stick around for each batch
# Shuffle is also key to generalization. instead of feeding one by one in order. 

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

"""for data in trainset:
  print(data)
  break

x, y = data[0][0],data[1][0]
print(y)

print(data[0][0].shape)""" 

# IT is in the shape of (1,28,28) so what we can do is: 
# plt.imshow(data[0][0].view(28,28))

### NETWORK ###

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 64) ##fc1 means first fully connected layer
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, 10)

  #how the data will flow through the network
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = (self.fc4(x))
    return F.log_softmax(x, dim=1) #dim is similar to axis from numpy.


net = Net()
#print(net)

X = torch.rand(28,28)
X = X.view(-1,28*28) 

#-1 specifies that this input will be of an unknown shape.
#print(X)
output = net(X)
#print(output)
#print(X)

#The optimizer's goal is to lower the loss by adjusting the weights / bias
# That is what net.parameters refers to. Everything that can be modified
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
  for data, target in trainset:
    net.zero_grad()
    output = net(data.view(-1, 28*28))
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
  
  #print(loss)

correct = 0
total = 0

with torch.no_grad():
  for X, y in testset:
    output = net(X.view(-1,28*28))
    
    

    for idx, i in enumerate(output):
      if torch.argmax(i) == y[idx]:
        correct+=1
      total += 1


print("Accuracy ", round(correct/total, 3))

plt.imshow(X[1].view(28,28))
plt.show()








    

