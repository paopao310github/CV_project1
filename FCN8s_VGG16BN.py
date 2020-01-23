# Convert VGG16BN to a FCN8s network

import math
import torch
import torch.nn as nn
from torchsummary import summary

class Block(nn.Module):
    def __init__(self, in_channels,out_chchannels, padding):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_chchannels, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_chchannels)
        self.relu1 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

def conv1x1(in_channels, out_chchannels):
    return nn.Conv2d(in_channels, out_chchannels, 1, bias=False)

def conv7x7(in_channels, out_chchannels):
    return nn.Conv2d(in_channels, out_chchannels, 7, bias=False)

def upsample(out_channels,scale):
    return nn.ConvTranspose2d(out_channels,out_channels,scale*2, stride=scale,bias=False)

def make_layers(in_channels, layer_list, padding_list):
    """
    in_channels: 3the number of input's channels
    layer_list: something like [64,64], padding_list: same length as layer_list
    """
    layers = []
    for v,p in zip(layer_list,padding_list):
        layers += [Block(in_channels, v, p)]
        in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list, padding_list=[1,1]):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list, padding_list)
        
    def forward(self, x):
        out = self.layer(x)
        return out

class VGG16BN(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, n_class=21):
        super(VGG16BN, self).__init__()
        # input: (N,3,224,224)
        self.layer1 = Layer(3, [64, 64], [100,1]) # padding=100 for the first layer
        # output: (N,64,422,422)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output: (N,64,211,211)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output: (N,128,105,105)
        self.layer3 = Layer(128, [256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output: (N,256,52,52)
        self.layer4 = Layer(256, [512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output: (N,512,26,26)
        self.layer5 = Layer(512, [512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output: (N,512,13,13)

        self.relu = nn.ReLU(inplace = True)

        self.conv6 = conv7x7(512,4096)
        self.conv7 = conv1x1(4096,4096)
        #self.classifier = conv1x1(4096,n_class)


        # prediction at different down-sampled levels
        self.pred_ds32 = conv1x1(4096,n_class) 
        self.pred_ds16 = conv1x1(512, n_class) 
        self.pred_ds8 = conv1x1(256, n_class) 

        self.scale_up2 = upsample(n_class,2) 
        self.scale_up8 = upsample(n_class,8) 

    def forward(self, x):
        f1 = self.pool1(self.layer1(x)) # output (N,64,211,211)
        f2 = self.pool2(self.layer2(f1)) # output (N,128,105,105)
        f3 = self.pool3(self.layer3(f2)) # output (N,256,52,52)
        f4 = self.pool4(self.layer4(f3)) # output (N,512,26,26)
        f5 = self.pool5(self.layer5(f4)) # output (N,512,13,13)

        h = self.relu(self.conv6(f5)) # output (N,4096,7,7)
        h = self.relu(self.conv7(h)) # output (N,4096,7,7)

        # prediction from new Conv layers (1/32), then upsample 2x
        h = self.pred_ds32(h) # output (N,n_class,7,7)
        h = self.scale_up2(h) # output (N,n_class,16,16)
        scale_ds32 = h # ds32->up2 = 1/16
        
        # prediction from maxpool4 output (1/16), then cropped to a good size
        h = self.pred_ds16(f4) # output (N,n_class,26,26)
        
        # difference is 26-16 = 10, therefore offset = 5
        pred_ds16 = h[:,:,5:5+scale_ds32.size()[2],5:5+scale_ds32.size()[3]] # cropped to (N,n_class,16,16)
        
        # combine two outputs with the same dimension, then upsample 2x
        h = pred_ds16 + scale_ds32 # output (N,n_class,16,16)
        h = self.scale_up2(h) # upsample 2x, output (N,n_class,34,34)
        scale_ds16 = h # ds16->up2 = 1/8
        
        # prediction from maxpool3 output (1/8), then cropped to a good size
        h = self.pred_ds8(f3) # output: (N,n_class,52,52)
        
        # difference is 52-34 = 18, therefore offset = 9
        pred_ds8 = h[:,:,9:9+scale_ds16.size()[2],9:9+scale_ds16.size()[3]] # cropped to (N,n_class,34,34)
        
        h = pred_ds8 + scale_ds16 # combine two outputs with the same dimension
        h = self.scale_up8(h) # upsample 8x, output (N,n_class,280,280)
 
        # difference is 280-224 = 56, therefore offset = 28
        h = h[:,:,28:28+scale_ds16.size()[2],28:28+scale_ds16.size()[3]] # Final output = same dimension as input
        
        return h


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16BN().to(device)
summary(model, (3,224,224))
