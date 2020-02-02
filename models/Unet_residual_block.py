# Use resnet block in a U-net

import torch.nn as nn
import torch
from torchsummary import summary

def conv1x1(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) 

def conv3x3(in_channels, out_channels, stride=1, padding=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)  

def upsample(in_channels,out_channels,scale):
    #return nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,scale*2, stride=scale),nn.BatchNorm2d(out_channels))
    return nn.ConvTranspose2d(in_channels,out_channels,scale*2,stride=scale)

def downsample(in_channels,out_channels,scale):
    return nn.MaxPool2d(kernel_size=scale, stride=scale)
    #if in_channels == out_channels:
    #    return nn.MaxPool2d(kernel_size=scale, stride=scale)
    #else:
    #    return nn.Sequential(nn.MaxPool2d(kernel_size=scale, stride=scale),conv1x1(in_channels,out_channels))

class ResBlock_encoder(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1,bottom=False):
        super(ResBlock_encoder, self).__init__()

        self.conv3a = conv3x3(in_channels, out_channels)
        self.conv3b = conv3x3(out_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.bottom = bottom
        if not self.bottom:
            self.conv1 = conv1x1(in_channels,out_channels)
            self.resize = downsample(out_channels,out_channels,2)
        else:
            self.conv1 = conv1x1(out_channels,in_channels)
            self.resize = upsample(in_channels,in_channels,2)

        #self.stride = stride

    def forward(self, x):
        identity = x

        # The 1st 3x3 convolutional layer
        out = self.conv3a(x)
        out = self.bn(out)
        out = self.relu(out)

        # The 2nd 3x3 convolutional layer
        out = self.conv3b(out)
        out = self.bn(out)

        # Save the output for the purpose of skip-connection
        skip = out

        if not self.bottom:
            out = self.resize(self.relu(self.conv1(identity) + out))
        else:
            out = self.resize(self.relu(identity + self.conv1(out)))

        return out,skip

class ResBlock_decoder(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1,top=False):
        super(ResBlock_decoder, self).__init__()

        self.conv3a = conv3x3(in_channels*2, in_channels)
        self.conv3b = conv3x3(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.top = top
        if not self.top:
            self.conv1 = conv1x1(in_channels*2,in_channels)
            self.resize = upsample(in_channels,out_channels,2) 
        else:
            self.conv1 = conv1x1(in_channels,out_channels)

    def _crop(self,x,skip): 
        _, _, h1, w1 = x.shape
        _, _, h2, w2 = skip.shape

        h0, w0 = min(h1,h2),min(w1,w2)

        dh1 = (h1 - h0) // 2 if h1 > h0 else 0
        dw1 = (w1 - w0) // 2 if w1 > w0 else 0
        dh2 = (h2 - h0) // 2 if h2 > h0 else 0
        dw2 = (w2 - w0) // 2 if w2 > w0 else 0
        return x[:, :, dh1: (dh1 + h0), dw1: (dw1 + w0)], \
                skip[:, :, dh2: (dh2 + h0), dw2: (dw2 + w0)]

    def forward(self,x,skip):
        _x, _skip = self._crop(x,skip)
        out = torch.cat([_x, _skip], dim=1)

        identity = out

        out = self.conv3a(out)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv3b(out)
        out = self.bn(out)

        if not self.top:
            out = self.resize(self.relu(self.conv1(identity) + out))
        else:
            out = self.relu(self.conv1(out))

        return out

class UNet(nn.Module):
    def __init__(self, in_channels=3,out_channels=1):
        super(UNet, self).__init__()
   
        self.encoder1 = ResBlock_encoder(in_channels,64)
        self.encoder2 = ResBlock_encoder(64,128)
        self.encoder3 = ResBlock_encoder(128,256)
        self.encoder4 = ResBlock_encoder(256,512)
        self.encoder5 = ResBlock_encoder(512,1024,bottom=True)

        self.decoder1 = ResBlock_decoder(512,256)
        self.decoder2 = ResBlock_decoder(256,128)
        self.decoder3 = ResBlock_decoder(128,64)
        self.decoder4 = ResBlock_decoder(64,out_channels,top=True)

    def forward(self, x):
        skips = []

        out,skip = self.encoder1(x)
        skips.append(skip)
        out,skip = self.encoder2(out)
        skips.append(skip)
        out,skip = self.encoder3(out)
        skips.append(skip)
        out,skip = self.encoder4(out)
        skips.append(skip)
        out,_ = self.encoder5(out)
        
        out = self.decoder1(out,skips[-1])
        out = self.decoder2(out,skips[-2])
        out = self.decoder3(out,skips[-3])
        out = self.decoder4(out,skips[-4])

        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
summary(model, (3,572,572))
