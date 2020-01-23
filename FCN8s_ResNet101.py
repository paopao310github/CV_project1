# Convert ResNet101 to a FCN8s network

import torch
import torch.nn as nn
from torchsummary import summary

def conv3x3(in_channels, out_channels, stride=1, padding=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)  

def conv1x1(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) 

def conv7x7(in_channels, out_chchannels):
    return nn.Conv2d(in_channels, out_chchannels, 7, bias=False)

def upsample(out_channels,scale):
    return nn.ConvTranspose2d(out_channels,out_channels,scale*2, stride=scale)

class BottleNeck(nn.Module):
    def __init__(self, in_channels,main_channels,downsample=None,stride=1):
        super(BottleNeck, self).__init__()

        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(in_channels, main_channels)
        self.bn1 = norm_layer(main_channels)
        self.conv2 = conv3x3(main_channels, main_channels, stride)
        self.bn2 = norm_layer(main_channels)
        self.conv3 = conv1x1(main_channels, main_channels*4) 
        self.bn3 = norm_layer(main_channels*4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
    
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_FCN8s(nn.Module):

    def __init__(self,n_class = 20):
        super(ResNet_FCN8s,self).__init__()

        self.norm_layer = nn.BatchNorm2d

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3,self.in_channels,kernel_size=7,stride=2,padding=102,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage1 = self.make_stage(64,3)
        self.stage2 = self.make_stage(128,4,stride=2)
        self.stage3 = self.make_stage(256,23,stride=2)
        self.stage4 = self.make_stage(512,3,stride=2)

        self.pred_ds32 = conv7x7(2048,n_class) 
        self.pred_ds16 = conv1x1(1024,n_class)
        self.pred_ds8 = conv1x1(512,n_class)

        self.scale_up2 = upsample(n_class,2) 
        self.scale_up8 = upsample(n_class,8) 

    def make_stage(self, main_channels, n_block, stride=1):
        downsample = None
        out_channels = main_channels * 4

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels,out_channels,stride),
                self.norm_layer(out_channels))

        stage = []
        stage.append(BottleNeck(self.in_channels,main_channels,downsample,stride))
        
        self.in_channels = out_channels

        for i in range(1, n_block):
           stage.append(BottleNeck(self.in_channels, main_channels))
        return nn.Sequential(*stage)  
       
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # output (N,64,211,211)

        f1 = self.stage1(x) # output (N,256,106,106)
        f2 = self.stage2(f1) # output (N,512,53,53)
        f3 = self.stage3(f2) # output (N,1024,27,27)
        f4 = self.stage4(f3) # output (N,2048,14,14)

        h = self.pred_ds32(f4)  # (N,n_class,8,8)
        h = self.scale_up2(h) # (N,n_class,18,18)
        scale_ds32 = h # ds32->up2 = 1/16

        h = self.pred_ds16(f3) # output (N,n_class,27,27)

        # difference is 27-18 = 9, therefore offset = 5
        pred_ds16 = h[:,:,5:5+scale_ds32.size()[2],5:5+scale_ds32.size()[3]] # cropped to (N,n_class,16,16)

        h = pred_ds16 + scale_ds32 # output (N,n_class,18,18)
        h = self.scale_up2(h) # upsample 2x, output (N,n_class,38,38)
        scale_ds16 = h # ds16->up2 = 1/8

        h = self.pred_ds8(f2) # output: (N,n_class,53,53)

        # difference is 53-38 = 15, therefore offset = 8
        pred_ds8 = h[:,:,8:8+scale_ds16.size()[2],8:8+scale_ds16.size()[3]] # cropped to (N,n_class,34,34)

        h = pred_ds8 + scale_ds16 # combine two outputs with the same dimension
        h = self.scale_up8(h) # upsample 8x, output (N,n_class,312,312)

        # difference is 312-224 = 88, therefore offset = 44
        h = h[:,:,44:44+scale_ds16.size()[2],44:44+scale_ds16.size()[3]] # Final output = same dimension as input

        return h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet_FCN8s().to(device)
summary(model, (3,224,224))
