import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    self.conv  = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)

class UNet(nn.Module):
    def __init__(
      self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], 
    ):
      super(UNet, self).__init__()
      # storing conv blocks of encoder and decoder
      self.downwards = nn.ModuleList()
      self.upwards = nn.ModuleList()
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

      # encoder
      for f in features:
        self.downwards.append(ConvBlock(in_channels, f))
        in_channels = f

      # latent space need 512 -> 1024
      self.latent = nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1)

      # decoder
      for f in features[::-1]:
        self.upwards.append(
            nn.ConvTranspose2d(
              # in_channels = f*2 because of concatenation from skip connection
              feature*2, feature, kernel_size=2, stride=2)
            )
        self.upwards.append(ConvBlock(feature*2, feature))
        in_channels = f
        
      # final layer
      self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
      # encoder
      encoder_outputs = []
      for block in self.downwards:
        x = block(x)
        encoder_outputs.append(x)
        x = self.pool(x)

      # latent space
      x = self.latent(x)
      x = x.view(x.size(0), -1, x.size(2), x.size(3))
      #skip_connection = encoder_outputs[-1]

      # decoder
      for block in self.upwards:
        x = F.relu(block(x))

      x = self.final(x)
      x = F.sigmoid(x)

      return x
