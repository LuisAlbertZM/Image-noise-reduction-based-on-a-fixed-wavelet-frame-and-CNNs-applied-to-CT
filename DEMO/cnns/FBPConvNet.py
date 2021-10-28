import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Sequential

"""
    Description: This class is a block of the encoding
    path
    Constructor parameters
        * indf: Number of input feature maps
        * ondf: Number of output feature maps
    Inputs
        * x: Input signal
    Outputs
        * down: Downsampled output signal
        * unde: Undecimated output signal
"""
class __unet_conv_block__(nn.Module):
    def __init__(self, indf, ondf):
        super(__unet_conv_block__, self).__init__()
        self.cblock1 = Sequential(
            nn.Conv2d(indf, ondf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ondf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ondf, ondf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ondf),
            nn.ReLU(inplace=True))
        self.cblock2 = Sequential(
            nn.ReflectionPad2d(padding =(0,1,0,1)),
            nn.MaxPool2d(kernel_size=2))
    def forward(self, x):
        unde = self.cblock1(x)
        down = self.cblock2(unde)
        return(down, unde)

    
"""
    Description: This class is a bloick of the decoding
    path
    Constructor parameters
        * indf: Number of input feature maps
        * ondf: Number of output feature maps
    Inputs
        * x: Input signal
        * bridge: Signal from skip-connection
    Outputs
        * conv: Output signal
"""
class __unet_up_block__(nn.Module):
    def __init__(self, indf, ondf, kernel_size=3, padding=1):
        super(__unet_up_block__, self).__init__()  
        self.reduce = nn.Sequential(
            nn.Conv2d(indf*2, indf, kernel_size=1, padding=0),
            nn.BatchNorm2d(indf),
            nn.ReLU(inplace=True))
        self.cblock = nn.Sequential(
            nn.Conv2d(indf  , ondf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ondf),
            nn.ReLU(inplace=True))
        self.up = nn.Sequential( 
            nn.ConvTranspose2d(
                indf, indf, kernel_size = 1,
                stride=2, output_padding=1, bias=True),
            nn.BatchNorm2d(indf),
            nn.ReLU(inplace=True))
    def forward(self, x, bridge):
        conc = torch.cat([self.up(x),bridge],1)
        red = self.reduce(conc)
        out = self.cblock(red)
        return(out)

"""
    Description: This class is an implementation of 
    FBPConvNet
    Constructor parameters
        * in_chans: Number of channels of input signal
        * depth: Number of decomposition levels
        * wf: Number of feature maps after the first
            decomposition level
    Inputs
        * x: Input signal
    Outputs
        * out: filtered image
"""
class FBPConvNet_2d(nn.Module):
    def __init__( self, in_chans=2, depth=5, wf=16):
        super(FBPConvNet_2d, self).__init__()
        self.depth = depth
        
        #  Filters for encoding section
        prev_channels = in_chans
        self.down_path = nn.ModuleList()
        for i in range(depth):
            out_chans = int(4*in_chans*(wf**(i+1)))
            self.down_path += [__unet_conv_block__( prev_channels, out_chans)]
            prev_channels = out_chans 
        self.up_path = nn.ModuleList()
        
        # Layers for decoding section
        for i in range(depth):
            out_chans = int(4*in_chans*(wf**(depth-i-1)))
            self.up_path += [__unet_up_block__( prev_channels, out_chans)]
            prev_channels = out_chans

        self.last = nn.Sequential(
            nn.Conv2d(prev_channels, in_chans, kernel_size=1))
        
    def forward(self, x):
        blocks = []
        bridges = []
        x_in = x
        
        # Encoding
        for i, down in enumerate(self.down_path):
            x, bridge = down(x)
            bridges.append(bridge)
        
        # Decoding
        for i, up in enumerate(self.up_path):
            ind = self.depth - i -1
            x = up(x, bridges[ind])
            
        out = x_in + self.last(x)
        return(out)