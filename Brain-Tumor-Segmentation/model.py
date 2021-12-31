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
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
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
      # 161 x 161 -> pool (80x80) -> upsample 160 x 160

      # encoder
      for f in features:
        self.downwards.append(ConvBlock(in_channels, f))
        in_channels = f

      # latent space need 512 -> 1024
      self.latent = ConvBlock(features[-1], features[-1]*2)

      # decoder
      for f in features[::-1]:
        self.upwards.append(
            nn.ConvTranspose2d(
              # in_channels = f*2 because of concatenation from skip connection
              f*2, f, kernel_size=2, stride=2)
            )
        self.upwards.append(ConvBlock(f*2, f))
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
      #x = x.view(x.size(0), -1, x.size(2), x.size(3))
      skip_connections = encoder_outputs[::-1]

      # decoder
      """for block in self.upwards:
        x = block(x)
        # concatenate with skip connection
        #x = torch.cat((x, skip_connection), 1)
        #skip_connection = x
      x = self.final(x)
      return x"""


      # decoder
      for layer in range(0, len(self.upwards), 2):
            x = self.upwards[layer](x)
            skip_connection = skip_connections[layer//2]

            # need output shape to match inital input shape so ensure at each step stays divisible by 16
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.upwards[layer+1](concat_skip)

      return self.final(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
torch.save(model.state_dict(), 'model.pth')"""

def testing():
  x = torch.randn(3,3,160,160)
  model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
  pred = model(x)
  print(pred.shape)
  print(x.shape)


