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
class shrink_dhsn1(nn.Module):
    def __init__(self, no_fmaps):
        super(shrink_dhsn1, self).__init__()
        # Defining the network
        self.convblock1 = nn.Sequential(
            nn.Conv2d(no_fmaps, no_fmaps, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU())
        self.convblock2 = nn.Sequential(
            nn.Conv2d(no_fmaps, no_fmaps, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU())
        self.convblock1res = nn.Sequential(
            nn.Conv2d(no_fmaps*2, no_fmaps, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU())
        self.convblock2res = nn.Sequential(
            nn.Conv2d(no_fmaps*2, no_fmaps, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, D):
        # Envelope detectiom
        envelope = torch.abs(D)
        
        conv1 = self.convblock1(envelope)
        convr1 = self.convblock1res(torch.cat([conv1, envelope],axis=1) )
        conv2 = self.convblock2(convr1)   
        convr2 = self.convblock2res(torch.cat([conv2, convr1],axis=1) )
        shrink = convr2*D
        return(shrink)


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
class dhsn1_2d(nn.Module):
    def __init__( self, in_channels=1, depth=1):
        super(dhsn1_2d, self).__init__()
        self.depth = depth

        # Encoding path
        self.shrink = nn.ModuleList()
        for i in range(depth):
            out_channels = int( in_channels*4**(i+1)) 
            self.shrink+= [shrink_dhsn1(out_channels)]
        
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