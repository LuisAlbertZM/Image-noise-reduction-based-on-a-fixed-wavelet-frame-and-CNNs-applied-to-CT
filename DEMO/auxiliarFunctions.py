import torch
import h5py
from torch.utils import data
import torch.nn as nn 
import torch.nn.functional as F

###########################################
# Single image dataset ####################
###########################################
class dataset_one_scan(data.Dataset):
    def __init__(self, path_low, path_ful, scan, level, width, valid=0):
        with h5py.File(path_low, 'r') as f:
            self.low =  f[scan][:]-24
        with h5py.File(path_ful, 'r') as f:
            self.ful = f[scan][:]-24
        
        self.no_slices = self.low.shape[1]    
        self.indexes = np.arange(self.no_slices)
        np.random.shuffle(self.indexes)
        
        # Level and window
        self.level = level
        self.width = width
        
        
    def __getitem__(self, i):
        indx = self.indexes[i]
        
        s = self.low[:,indx,:].shape
        
        a_min = self.level-self.width/2
        a_max = self.level+self.width/2
        
        img_low = torch.from_numpy(self.low[:,indx,:]
            ).view(1,1,s[0],s[1]).float().cuda()
        img_ful = torch.from_numpy(self.ful[:,indx,:]
            ).view(1,1,s[0],s[1]).float().cuda()
        
        in_window = ((self.low[:,indx,:] > a_min)*(self.ful[:,indx,:] < a_max))
        in_window = torch.from_numpy(in_window).view(1,1,s[0],s[1]).float().cuda()
        
        return([img_low, img_ful])
    
    def __len__(self):
        return(self.no_slices)
    

class gini_index(torch.nn.Module):
    def __init__(self):
        super(gini_index, self).__init__()
    
    def forward(self, sig):
        sig = sig + 1e-20
        sort, _ = torch.sort( torch.abs(sig.view(-1)), descending=False )
        N = sort.numel()
        ks =torch.arange(N, dtype = sig.dtype ,device=sig.device)
        GI = 2*torch.sum( (sort/(torch.sum(sort)))*( (N - ks - 0.5)/(N)) )
        return(GI)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class iwL1(nn.Module):
    def __init__(self, level, width):
        super(iwL1, self).__init__()
        self.vmax =  level + width/2
        self.vmin =  level - width/2
    def forward(self, x, gt):
        with torch.no_grad():
            iw = (( self.vmin < gt)*(gt < self.vmax)).to(torch.float32)
        l1 = torch.abs(gt-x)
        return( torch.sum(iw*l1)/(torch.sum(iw)+1) )
    
class dataset_one_scan_test(data.Dataset):
    def __init__(self, path_low, path_ful, scan):
        with h5py.File(path_low, 'r') as f:
            self.low =  f[scan][:]-24
        with h5py.File(path_ful, 'r') as f:
            self.ful = f[scan][:]-24
        
        self.no_slices = self.low.shape[1]
        
    def __getitem__(self, i):
        
        s = self.low[:,i,:].shape
        
        img_low = torch.from_numpy(
            self.low[:,i,:]
            ).view(1,1,s[0],s[1]).float().cuda()
        img_ful = torch.from_numpy(
            self.ful[:,i,:]
            ).view(1,1,s[0],s[1]).float().cuda()
        
        return([img_low, img_ful])
    
    def __len__(self):
        return(self.no_slices)
    

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
def imgs_w_zoom(dataset_red, dataset_hig, gen, sl, test_scan, level, width, xc, yc, of):
    
    a_min = level-width/2
    a_max = level+width/2
    
    # Loading scan
    sc =  dataset_one_scan_test(dataset_red, dataset_hig, test_scan )
    low_im, ful_im = sc.__getitem__(sl)
    with torch.no_grad():
        inp = low_im.detach().cpu().numpy()[0,0,:,:]
        gtr = ful_im.detach().cpu().numpy()[0,0,:,:]
        ofs = 128
        evalu = gen( F.pad(low_im,(ofs,ofs,ofs,ofs),"reflect")
                )[0,0,ofs:-ofs,ofs:-ofs] 
        evalu = evalu.cpu().detach().numpy()
        res = evalu

    fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(24,9.5))

    # Input
    axs[0].imshow(inp[:,:], vmax = a_max, vmin = a_min,cmap="gray")
    axs[0].yaxis.set_major_locator(plt.NullLocator())
    axs[0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0].add_patch(patches.Rectangle((xc,yc),of,of,linewidth=2,edgecolor='r',facecolor='none'))
    axins = axs[0].inset_axes([0.65, 0.0, 0.35, 0.35])
    axins.imshow(np.flipud(inp[yc:yc+of,xc:xc+of]), vmax = a_max, vmin = a_min,origin="lower",cmap="gray")
    axins.yaxis.set_major_locator(plt.NullLocator())
    axins.xaxis.set_major_formatter(plt.NullFormatter())


    # Result
    axs[1].imshow(res[:,:], vmax = a_max, vmin = a_min,cmap="gray")
    axs[1].yaxis.set_major_locator(plt.NullLocator())
    axs[1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1].add_patch(patches.Rectangle((xc,yc),of,of,linewidth=2,edgecolor='r',facecolor='none'))
    axins1 = axs[1].inset_axes([0.65, 0.0, 0.35, 0.35])
    axins1.imshow(np.flipud(res[yc:yc+of,xc:xc+of]), vmax = a_max, vmin = a_min, origin="lower",cmap="gray")
    axins1.yaxis.set_major_locator(plt.NullLocator())
    axins1.xaxis.set_major_formatter(plt.NullFormatter())
    axins2 = axs[1].inset_axes([0.65, 0.6, 0.35, 0.35])
    im = axins2.imshow(np.flipud(np.abs(res-gtr)[yc:yc+of,xc:xc+of]), vmax = a_max, vmin = a_min, origin="lower",cmap="gray")
    axins2.yaxis.set_major_locator(plt.NullLocator())
    axins2.xaxis.set_major_formatter(plt.NullFormatter())
    divider = make_axes_locatable(axins2)
    cax = divider.append_axes("left", size="5%", pad=0.0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15, labelcolor="y")  # set your label size here
    

    
    # Ground Truth
    axs[2].imshow(gtr[:,:], vmax = a_max, vmin = a_min,cmap="gray")
    axs[2].yaxis.set_major_locator(plt.NullLocator())
    axs[2].xaxis.set_major_formatter(plt.NullFormatter())
    axs[2].add_patch(patches.Rectangle((xc,yc),of,of,linewidth=2,edgecolor='r',facecolor='none'))
    axins = axs[2].inset_axes([0.65, 0.0, 0.35, 0.35])
    axins.imshow(np.flipud(gtr[yc:yc+of,xc:xc+of]), vmax = a_max, vmin = a_min, origin="lower",cmap="gray")
    axins.yaxis.set_major_locator(plt.NullLocator())
    axins.xaxis.set_major_formatter(plt.NullFormatter())
    
    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.show()

    
"""
    Radial profile
    Obtained from 
        Thread: What is the best way to calculate
            radial average of the image with python?
        Link: https://stackoverflow.com/questions/...
            48842320/...
            what-is-the-best-way-to-calculate-radial-average-of-the-image-with-python
"""
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


