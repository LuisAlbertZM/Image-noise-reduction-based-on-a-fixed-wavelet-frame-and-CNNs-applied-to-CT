import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F


"""
    Description: This class upsamples an image by inserting
    zeros
    Inputs
        * x: Image to be upsampled by pading with zeros
    Outputs
        * y: Upsampled image
"""
class __upsample_zeropadding_2D__(nn.Module):
    def __init__(self):
        super(__upsample_zeropadding_2D__, self).__init__()
        
        # Fixed unity kernel
        k_np = np.asanyarray([1])
        self.k = nn.Parameter(data = torch.from_numpy(k_np),
            requires_grad=False).float().cuda().reshape((1,1,1,1))
        
    def forward(self, x):
        xs = x.shape
        x_p = x.view(xs[0]*xs[1], 1, xs[2], xs[3])

        # Upsampling with a transposed convolution with unity
        # kernel
        up = F.conv_transpose2d(x_p, weight=self.k, stride=(2,2), dilation=1)
            
        # Ensuring that the output is twice as large as the input
        if up.shape[2] < x.shape[2]*2:
            up = F.pad(input = up, pad = (0, 0, 0, 1), mode="reflect")
        if up.shape[3] < x.shape[3]*2:
            up = F.pad(input = up, pad = (0, 1, 0, 0), mode="reflect")
        
        us = up.shape 
        return(up.view(xs[0], xs[1], us[2], us[3]))

    
"""
    Description: Compact implementation of forward Overcomplete
    Haar Wavelet Transform
    Inputs
        * x: Image to be decomposed
    Outpus
        * LL: Low-frequency band
        * LH: Horizontal high-frequency band
        * HL: Vertical high-frequency band
        * HH: Diagonal high-frequency band
"""
class fohwt(nn.Module):
    def __init__( self, undecimated=False, mode="reflect"):
        super(fohwt, self).__init__()
        self.mode = mode
        #self.no_chans
        if undecimated: self.stride=1
        else: self.stride=2
    
        # 2D Overcomplete Haar Wavelet Transform
        LL = np.asarray([[
            [[ 1.,  1.,  0.], [ 1.,  2.,  1.], [ 0.,  1.,  1.]],
            [[-1., -1.,  0.], [-1.,  0.,  1.], [ 0.,  1.,  1.]],
            [[ 0.,  1.,  1.], [ 1.,  2.,  1.], [ 1.,  1.,  0.]],
            [[ 0., -1., -1.], [ 1.,  0., -1.], [ 1.,  1.,  0.]]]
            ])/(2*np.sqrt(2))
        LH = np.asarray([[
            [[-1.,  1.,  0.], [-1.,  0.,  1.], [ 0., -1.,  1.]],
            [[ 1., -1.,  0.], [ 1., -2.,  1.], [ 0., -1.,  1.]],
            [[ 0., -1.,  1.], [-1.,  0.,  1.], [-1.,  1.,  0.]],
            [[ 0.,  1., -1.], [-1.,  2., -1.], [-1.,  1.,  0.]]]
            ])/(2*np.sqrt(2))
        HL = np.asarray([[
            [[-1., -1.,  0.], [ 1.,  0., -1.], [ 0.,  1.,  1.]],
            [[ 1.,  1.,  0.], [-1., -2., -1.], [ 0.,  1.,  1.]],
            [[ 0., -1., -1.], [-1.,  0.,  1.], [ 1.,  1.,  0.]],
            [[ 0.,  1.,  1.], [-1., -2., -1.], [ 1.,  1.,  0.]]]
            ])/(2*np.sqrt(2))
        HH = np.asarray([[
            [[ 1., -1.,  0.], [-1.,  2., -1.], [ 0., -1.,  1.]],
            [[-1.,  1.,  0.], [ 1.,  0., -1.], [ 0., -1.,  1.]],
            [[ 0.,  1., -1.], [ 1., -2.,  1.], [-1.,  1.,  0.]],
            [[ 0., -1.,  1.], [ 1.,  0., -1.], [-1.,  1.,  0.]]
            ]])/(2*np.sqrt(2))
        
        # Converting to parameters
        self.LL = nn.Parameter(
            torch.from_numpy(LL), requires_grad=False).to(torch.float32).cuda()
        self.LH = nn.Parameter(
            torch.from_numpy(LH), requires_grad=False).to(torch.float32).cuda()
        self.HL = nn.Parameter(
            torch.from_numpy(HL), requires_grad=False).to(torch.float32).cuda()
        self.HH = nn.Parameter(
            torch.from_numpy(HH), requires_grad=False).to(torch.float32).cuda()
    
    def forward(self,x):
        # Loading parameters
        mode = self.mode
        stride = self.stride

        # Reshaping for easy convolutions
        xp = F.pad(x, (1,1,1,1), mode=mode, value=0)
        xps = xp.shape
        xpp = xp.view(xps[0]*xps[1], 1, xps[2], xps[3])
        
        # Convolving with the sub-kernels
        LL = F.conv2d(xpp, self.LL.transpose(1,0), bias=None, stride=stride)
        LH = F.conv2d(xpp, self.LH.transpose(1,0), bias=None, stride=stride)
        HL = F.conv2d(xpp, self.HL.transpose(1,0), bias=None, stride=stride)
        HH = F.conv2d(xpp, self.HH.transpose(1,0), bias=None, stride=stride)
            
        osi = [xps[0],xps[1]*4, LL.shape[2], LL.shape[3]]
        return([LL.view(*osi), LH.view(*osi), HL.view(*osi), HH.view(*osi)])

    
"""
    Description: Compact implementation of inverse Overcomplete
    Haar Wavelet Transform
    Inputs
        * LL: Low-frequency band
        * LH: Horizontal high-frequency band
        * HL: Vertical high-frequency band
        * HH: Diagonal high-frequency band
    Outpus
        * x: Image to be decomposed
"""
class iohwt(nn.Module):
    def __init__( self, undecimated=False,mode="reflect"):
        super(iohwt, self).__init__()
        self.mode = mode
        self.undecimated = undecimated
        self.upsample = __upsample_zeropadding_2D__()
        
        # Inverse diagonal Haar DWT
        if undecimated: k=16
        else: k=4
            
        iLL = np.asarray([[
            [[ 1.,  1.,  0.], [ 1.,  2.,  1.], [ 0.,  1.,  1.]],
            [[ 1.,  1.,  0.], [ 1.,  0., -1.], [ 0., -1., -1.]],
            [[ 0.,  1.,  1.], [ 1.,  2.,  1.], [ 1.,  1.,  0.]],
            [[ 0.,  1.,  1.], [-1.,  0.,  1.], [-1., -1.,  0.]]
            ]])/(2*k*np.sqrt(2))
        iLH = np.asarray([[
            [[ 1., -1.,  0.], [ 1.,  0., -1.], [ 0.,  1., -1.]],
            [[ 1., -1.,  0.], [ 1., -2.,  1.], [ 0., -1.,  1.]],
            [[ 0.,  1., -1.], [ 1.,  0., -1.], [ 1., -1.,  0.]],
            [[ 0.,  1., -1.], [-1.,  2., -1.], [-1.,  1.,  0.]]
            ]])/(2*k*np.sqrt(2))
        iHL = np.asarray([[
            [[ 1.,  1.,  0.], [-1.,  0.,  1.], [ 0., -1., -1.]],
            [[ 1.,  1.,  0.], [-1., -2., -1.], [ 0.,  1.,  1.]],
            [[ 0.,  1.,  1.], [ 1.,  0., -1.], [-1., -1.,  0.]],
            [[ 0.,  1.,  1.], [-1., -2., -1.], [ 1.,  1.,  0.]]
            ]])/(2*k*np.sqrt(2))
        iHH = np.asarray([[
            [[ 1., -1.,  0.], [-1.,  2., -1.], [ 0., -1.,  1.]],
            [[ 1., -1.,  0.], [-1.,  0.,  1.], [ 0.,  1., -1.]],
            [[ 0.,  1., -1.], [ 1., -2.,  1.], [-1.,  1.,  0.]],
            [[ 0.,  1., -1.], [-1.,  0.,  1.], [ 1., -1.,  0.]]
            ]])/(2*k*np.sqrt(2))
        self.iLL = nn.Parameter(
            torch.from_numpy(iLL), requires_grad=False).to(torch.float32).cuda()
        self.iLH = nn.Parameter(
            torch.from_numpy(iLH), requires_grad=False).to(torch.float32).cuda()
        self.iHL = nn.Parameter(
            torch.from_numpy(iHL), requires_grad=False).to(torch.float32).cuda()
        self.iHH = nn.Parameter(
            torch.from_numpy(iHH), requires_grad=False).to(torch.float32).cuda()
    
    def forward(self, LL, LH, HL, HH):
        # Loading parameters
        mode=self.mode
        stride=1

        # Upsampling by inserting zeros
        x = torch.cat([LL, LH, HL, HH],axis=1)
        if self.undecimated: xp = x
        else: xp = self.upsample(x)
        xpp = F.pad(xp,(1,1,1,1),mode=mode,value=0)
        LL2, LH2, HL2, HH2 = torch.split(xpp, LL.shape[1], dim=1)

        lls = LL2.shape
        # Inverse transform
        cs = [lls[0]*lls[1]//4,4,lls[2],lls[3]]
        iLL = F.conv2d(LL2.reshape(*cs), self.iLL, bias=None, stride=stride)
        iLH = F.conv2d(LH2.reshape(*cs), self.iLH, bias=None, stride=stride)
        iHL = F.conv2d(HL2.reshape(*cs), self.iHL, bias=None, stride=stride)
        iHH = F.conv2d(HH2.reshape(*cs), self.iHH, bias=None, stride=stride)
        
        xiw = iLL+iLH+iHL+iHH
        xiws = xiw.shape
        
        return( xiw.view( lls[0], lls[1]//4, xiws[2], xiws[3])) 