# Some classes copied from
# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py

import torch
from torch import nn
from torch.nn import functional as F

# Copied from pytorch code
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# Copied from pytorch code
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# Copied from pytorch code
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

#------------------------------------------------------------------------------
def upsample(x,scale):
    C, H, W = x.shape[-3:]
    return F.interpolate(x, size=(C,H*scale,W*scale), mode='bilinear', align_corners=False)

class DCNNconv(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,padding=1,dilation=1,stride=1,pooling=False):
        super(DCNNconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            padding=padding, dilation=dilation, stride=stride,bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.Relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=2,stride=2)

        self.pooling = pooling

    def forward(self,x):
        x = self.Relu(self.BN(self.conv(x)))
        if self.pooling: x = self.maxpooling(x)
        return x
        

class DCNN(nn.Module):
    def __init__(self, in_channels):
        super(DCNN, self).__init__()

        self.block1 = DCNNconv(in_channels, 64, stride=2,pooling=True) # 1/4
        self.block2 = DCNNconv(64,  128, pooling=True)  # 1/8 
        self.block3 = DCNNconv(128, 256, pooling=True) # 1/16
        self.block4 = DCNNconv(256, 256, dilation=2)  # 1/16
        self.block5 = DCNNconv(256, 256, dilation=4)  # 1/16
        self.block6 = DCNNconv(256, 256, dilation=8)  # 1/16
        self.block7 = DCNNconv(256, 256, dilation=16) # 1/16

    def forward(self,x):
        x = self.block1(x)
        llf = x # low level feature
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return x, llf

class Decoder(nn.Module):
    def __init__(self, in_channels,num_class):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels//4,in_channels,1)
        self.conv3 = nn.Conv2d(in_channels, num_class, 3, stride=1,padding=1)

    def forward(self,LLF,ASPP_output):
        x = self.conv1(LLF)
        skip = upsample(ASPP_output, scale=4)
        x = torch.cat([x,skip],dim=1)
        x = self.conv3(x)
        x = upsample(x, scale=4)
    
        return x

class Deeplabv3p(nn.Module):
    def __init__(self,in_channels,num_class):
        super(Deeplabv3p,self)
        self.DCNN = DCNN(in_channels)
        self.ASPP = ASPP(256, [6,12,18])
        self.Decoder = Decoder(256,num_class)

    def forward(self,x):
        x,LLF = self.DCNN(x)
        ASPP_output = self.APSS(x)
        x = self.Decoder(LLF,ASPP_output)

        return x
