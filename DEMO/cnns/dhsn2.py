import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Sequential
from .ohwt import fohwt, iohwt

"""
    Description: This class is the thresholding CNN
    Constructor parameters
        * no_fmaps: Number of feature maps in which wavelet shrinkage
        is applied
    Inputs
        * D: detail/feature maps in which the shinkage is applied
    Outputs
        * shrink: Feature maps in which wavelet shrinkage has been applied
"""
class shrink_dhsn2(nn.Module):
    def __init__(self, no_fmaps):
        super(shrink_dhsn2, self).__init__()
        self.w_1 = nn.Parameter(
            0.5*torch.rand((no_fmaps, no_fmaps, 1, 1)) )
        
        self.w_2 = nn.Parameter(
            0.5*torch.rand((no_fmaps, no_fmaps, 1, 1)) )
        
        self.wp_1 = nn.Parameter(
            0.5*torch.rand((2*no_fmaps, 2*no_fmaps, 1, 1)) )
        self.bi_1 = nn.Parameter( torch.rand((2*no_fmaps)) )
        
        self.wp_2 = nn.Parameter(
            0.5*torch.rand((no_fmaps, 2*no_fmaps, 1, 1)) )
        self.bi_2 = nn.Parameter( torch.rand((no_fmaps)) )

    def forward(self, D):
        eps = 1e-4
        w_1 = F.relu(self.w_1)
        w_2 = F.relu(self.w_2)
        
        # Local power
        with torch.no_grad():
            ds = D.shape
            env1 = torch.abs(D)
            env2 = torch.sqrt(
                F.avg_pool2d(D**2, stride=1, kernel_size=3,padding=1))
            rat1 = 1/(env1+eps)
            rat2 = 1/(env2+eps)
        
        # Neighshrink-inspired section
        shrink1 = F.relu( 1 - F.conv2d(
            rat1, w_1, stride=1, padding=0, dilation=1, groups=1) )
        shrink2 = F.relu( 1 - F.conv2d(
            rat2, w_2, stride=1, padding=0, dilation=1, groups=1) )
        
        # Two-layer perceptron
        shrinkc = torch.cat([shrink1, shrink2], axis=1)
        p_1 = torch.relu(F.conv2d(shrinkc, self.wp_1, self.bi_1, stride=1, padding=0))
        p_2 = torch.sigmoid(F.conv2d(p_1, self.wp_2, self.bi_2, stride=1, padding=0))
        est = p_2*D
        
        return( est  )


"""
    Description: This class is the thresholding CNN
    Constructor parameters
        * no_fmaps: Number of feature maps in which wavelet shrinkage
        is applied
    Inputs
        * D: detail/feature maps in which the shinkage is applied
    Outputs
        * shrink: Feature maps in which wavelet shrinkage has been applied
"""
class dhsn2_2d(nn.Module):
    def __init__( self, in_channels=1, depth=1):
        super(dhsn2_2d, self).__init__()
        self.depth = depth

        # Encoding path
        self.shrink = nn.ModuleList()
        for i in range(depth):
            out_channels = int( in_channels*4**(i+1)) 
            self.shrink+= [shrink_dhsn2(out_channels)]
        
        # Defining inverse and forward transforms
        self.dwt = fohwt()
        self.idwt = iohwt()

    def forward(self, x, bypass_shrinkage=False):
        xs = x.shape
        LH_list = []; HL_list = []; HH_list = []
        
        # Forward transform
        for i in range(self.depth): 
            LL, LH, HL, HH = self.dwt(x)
            # Path bypass/shrink
            if bypass_shrinkage:
                LH_list.append(LH)
                HL_list.append(HL)
                HH_list.append(HH)
            else:
                LH_list.append(self.shrink[i]( LH ) )
                HL_list.append(self.shrink[i]( HL ) )
                HH_list.append(self.shrink[i]( HH ) )
            x = LL
        
        # Decoding section
        for i in range(self.depth):         
            indx = self.depth - i -1
            LL = self.idwt(LL, LH_list[indx], HL_list[indx], HH_list[indx])
            
        return(LL)