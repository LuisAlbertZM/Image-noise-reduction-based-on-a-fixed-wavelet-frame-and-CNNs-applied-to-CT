import os
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

""" Function: psnr
    Description:
        * This function computes the PSNR
    Inputs:
        * sc: input image
        * sc_ref: reference image
    Outputs:
        * None
"""
def psnr(sc, sc_ref, vmax):
    return( 10*np.log10( vmax**2/ np.mean((sc-sc_ref)**2) ) )

""" Function: slice_ssim
    Description:
        * This function computes the Mean Structural Similarity Index
    Inputs:
        * sc: input image
        * sc_ref: reference image
    Outputs:
        * None
"""
def slice_ssim(sc, sc_ref):
    mssim, grad, S = ssim(
        sc[:,:], sc_ref[:,:],
        gradient=True, full=True)    
    return( mssim )

""" Function: evalCNN
    Description: This function executes noise reduction on
        the signal x with a given cnn.
    inputs:
        * qdct: Signal to be filred with a ggiven cnn
        * cnn: network that performs noise reduction
    outputs:
        * eval_noHU: denoied signal
"""
def evalCNN(qdct, cnn):
    p=128
    # Evaluating the image. We use no_grad() to avoid
    # wasting memory on gradients, since we only perform 
    # forward passes
    with torch.no_grad():
        # We pad before feeding the signal
        # to mitigate the effects of zero-padding within
        # the CNN
        inp = torch.from_numpy(
            qdct).unsqueeze(0).unsqueeze(0).to(torch.float).cuda() 
        inpp = F.pad(inp,(p,p,p,p),"reflect")
        eva = cnn( inpp )
        eva2 = eva.cpu().detach().numpy()[0,0,p:-p,p:-p]
    return(eva2)


""" Function: displayRowCNNComparison
    Description:
        * This function computes the Mean Structural Similarity Index
    Inputs:
        * fdct, qdct, dhsn2_res, dhsn1_res, fbp_res, red_res: input images
        * xc, yc: Left-upper corner of zoomed section
    Outputs:
        * None
"""
def plot_zoom_and_diff(axs, proc, ref, xc, yc, of, vmin, vmax, zoom, isRef=False):
    labelsize = 15
    labelcolor="y"
    
    # Global plot
    axs.imshow( proc[:,:] ,cmap="gray",vmin=vmin, vmax=vmax)
    axs.yaxis.set_major_locator(plt.NullLocator())
    axs.xaxis.set_major_formatter(plt.NullFormatter())
    if not isRef:
        axs.add_patch(patches.Rectangle((xc,yc),of,of,linewidth=2,edgecolor='r',facecolor='none'))
    else:
        axs.add_patch(patches.Rectangle((xc,yc),of,of,linewidth=2,edgecolor='c',facecolor='none'))
        
    # Zoom
    inset1 = zoomed_inset_axes(axs, zoom=zoom, loc='lower left')
    img = inset1.imshow( proc[yc:yc+of,xc:xc+of] ,cmap="gray",vmin=vmin, vmax=vmax)
    inset1.yaxis.set_major_locator(plt.NullLocator())
    inset1.xaxis.set_major_locator(plt.NullLocator())
    if not isRef:
        inset1.patch.set_edgecolor('red')
        inset1.patch.set_linewidth('5')  
        # Difference zoom
        dif = np.abs(proc-ref)
        inset2 = zoomed_inset_axes(axs, zoom=zoom, loc='upper left')
        img = inset2.imshow( dif[yc:yc+of,xc:xc+of] ,cmap="hot", vmin=3, vmax=15)
        inset2.yaxis.set_major_locator(plt.NullLocator())
        inset2.xaxis.set_major_locator(plt.NullLocator())
        insetI = inset_axes(inset2, width="10%",height="100%", loc='lower right', bbox_to_anchor=(0.0, 0., 1, 1),
                         bbox_transform=inset2.transAxes, borderpad=0)
        cb = plt.colorbar(img, cax=insetI)
        cb.ax.tick_params(labelsize=labelsize, labelcolor=labelcolor)
    else:
        inset1 = zoomed_inset_axes(axs, zoom=zoom, loc='lower left')
        img = inset1.imshow( ref[yc:yc+of,xc:xc+of] ,cmap="gray",vmin=vmin, vmax=vmax)
        inset1.yaxis.set_major_locator(plt.NullLocator())
        inset1.xaxis.set_major_locator(plt.NullLocator())
        inset1.patch.set_edgecolor('cyan')
        inset1.patch.set_linewidth('5')

        
        
""" Function: displayRowCNNComparison
    Description:
        * This function computes the Mean Structural Similarity Index
    Inputs:
        * sli, ref: input images
        * ax: Axis where the image is displayed
        * xc, yc: Left-upper corner of zoomed section
        * meas: Display measurements of PSNR and MSSIM measurements
        * isRef: If true the difference image is not displayed
    Outputs:
        * None
"""
def displaySlice(sli, ref, ax, xc, yc, of, zoom, vmin, vmax, meas=True, isRef= False):
    # Making the subfigures
    plot_zoom_and_diff(ax, sli, ref, xc, yc, of, vmin, vmax, zoom, isRef)
    if meas:
        ax.text(180, 440, "MSSIM=%1.2f"%(
            slice_ssim(sli[yc:yc+of,xc:xc+of], ref[yc:yc+of,xc:xc+of])),
                    fontsize=13, color='orange',backgroundcolor='0.0' ) 

        ax.text(180, 490, "PSNR=%1.1f [dB]"%(
            psnr(sli[yc:yc+of,xc:xc+of], ref[yc:yc+of,xc:xc+of], vmax)),
                    fontsize=13, color='orange',backgroundcolor='0.0' ) 
    
    
""" Function: displayRowCNNComparison
    Description:
        * This function computes the Mean Structural Similarity Index
    Inputs:
        * fdct, qdct, dhsn2, dhsn1, fbp, red: input images
        * xc, yc: Left-upper corner of zoomed section
    Outputs:
        * None
"""
def displayRowCNNNoData(
    fdct, qdct, dhsn2, xc, yc, of, axs, zoom, vmin, vmax):
    displaySlice(qdct,  fdct, axs[0], xc, yc, of, zoom, vmin, vmax)
    displaySlice(dhsn2, fdct, axs[1], xc, yc, of, zoom, vmin, vmax)
    displaySlice(fdct,  fdct, axs[2], xc, yc, of, zoom, vmin, vmax, meas=False, isRef= True)
    

""" Function: displayRowCNNComparison
    Description:
        * This function computes the Mean Structural Similarity Index
    Inputs:
        * fdct, qdct, dhsn2, dhsn1, fbp, red: input images
        * xc, yc: Left-upper corner of zoomed section
    Outputs:
        * None
"""
def displayRowCNNComparison(
    fdct, qdct, dhsn2, dhsn1, fbp, red, xc, yc, of, axs, zoom, vmin, vmax):
    displaySlice(qdct,  fdct, axs[0], xc, yc, of, zoom, vmin, vmax)
    displaySlice(dhsn2, fdct, axs[1], xc, yc, of, zoom, vmin, vmax)
    displaySlice(dhsn1, fdct, axs[2], xc, yc, of, zoom, vmin, vmax)
    displaySlice(fbp,   fdct, axs[3], xc, yc, of, zoom, vmin, vmax)
    displaySlice(red,   fdct, axs[4], xc, yc, of, zoom, vmin, vmax)
    displaySlice(fdct,  fdct, axs[5], xc, yc, of, zoom, vmin, vmax, meas=False, isRef= True)

""" Function: figureNoiseReductionCNNComparison
    Description:
        * This function performs Fig. 8 of the paper which compares
            the noise-reduction performance
    Inputs:
        * dataDir: Input images which are shown in Fig. 8 
        * dhsn2, dhsn1, fbp, red: CNNs to be compared 
    Outputs:
        * None
"""
def figureNoiseReductionCNNComparison(dataDir, dhsn2, dhsn1, fbp, red):
    zoom=2
    vmin=0
    vmax=80

    # Making the figure
    sc=3
    fig, axs = plt.subplots(nrows=5, ncols=6,figsize=(sc*6,sc*5.11))


    dispDat = {
        0:{"Name":"N051", "xc":340, "yc":200, "of": 80,
           "axs":[axs[0,0], axs[0,1], axs[0,2], axs[0,3], axs[0,4], axs[0,5]]},
        1:{"Name":"N188", "xc":340, "yc":180, "of": 80,
           "axs":[axs[1,0], axs[1,1], axs[1,2], axs[1,3], axs[1,4], axs[1,5]]},
        2:{"Name":"N198", "xc":270, "yc":315, "of": 80,
           "axs":[axs[2,0], axs[2,1], axs[2,2], axs[2,3], axs[2,4], axs[2,5]]},
        3:{"Name":"N138", "xc":255, "yc":135, "of": 80,
           "axs":[axs[3,0], axs[3,1], axs[3,2], axs[3,3], axs[3,4], axs[3,5]]},
        4:{"Name":"N153", "xc":250, "yc":250, "of": 80,
           "axs":[axs[4,0], axs[4,1], axs[4,2], axs[4,3], axs[4,4], axs[4,5]]},
    }

    for i in np.arange(5):
        # Loading the data
        with h5py.File(dataDir, 'r') as f:
            qdct = f["%s_qdct"%(dispDat[i]["Name"])][:]
            fdct = f["%s_fdct"%(dispDat[i]["Name"])][:]

        # Evaluating the CNNs with qdct
        dhsn2_res = evalCNN(qdct, dhsn2)
        dhsn1_res = evalCNN(qdct, dhsn1)
        fbp_res = evalCNN(qdct, fbp)
        red_res = evalCNN(qdct, red)

        # Displaying
        displayRowCNNComparison(
            fdct, qdct, dhsn2_res, dhsn1_res, fbp_res, red_res,
            dispDat[i]["xc"], dispDat[i]["yc"], dispDat[i]["of"],
            dispDat[i]["axs"],
            zoom, vmin, vmax)
        if i==0:
            fs = 20
            dispDat[i]["axs"][0].set_title("QDCT", fontsize= fs)
            dispDat[i]["axs"][1].set_title("DHSN2", fontsize= fs)
            dispDat[i]["axs"][2].set_title("DHSN1", fontsize= fs)
            dispDat[i]["axs"][3].set_title("FPBConvNet", fontsize= fs)
            dispDat[i]["axs"][4].set_title("RED", fontsize= fs)
            dispDat[i]["axs"][5].set_title("FDCT", fontsize= fs)

    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.show()
    
""" Function: figureNoiseReductionWithoutData
    Description:
        * This function performs Fig. 8 of the paper which compares
            the noise-reduction performance
    Inputs:
        * dataDir: Input images which are shown in Fig. 8 
        * dhsn2_noData: CNNs to be compared 
    Outputs:
        * None
"""
def figureNoiseReductionWithoutData(dataDir, dhsn2_noData):
    zoom=2
    vmin=0
    vmax=80

    # Making the figure
    sc=3
    fig, axs = plt.subplots(nrows=5, ncols=3,figsize=(sc*3,sc*5.11))


    dispDat = {
        0:{"Name":"N051", "xc":340, "yc":200, "of": 80,
           "axs":[axs[0,0], axs[0,1], axs[0,2]]},
        1:{"Name":"N188", "xc":340, "yc":180, "of": 80,
           "axs":[axs[1,0], axs[1,1], axs[1,2]]},
        2:{"Name":"N198", "xc":270, "yc":315, "of": 80,
           "axs":[axs[2,0], axs[2,1], axs[2,2]]},
        3:{"Name":"N138", "xc":255, "yc":135, "of": 80,
           "axs":[axs[3,0], axs[3,1], axs[3,2]]},
        4:{"Name":"N153", "xc":250, "yc":250, "of": 80,
           "axs":[axs[4,0], axs[4,1], axs[4,2]]},
    }

    for i in np.arange(5):
        # Loading the data
        with h5py.File(dataDir, 'r') as f:
            qdct = f["%s_qdct"%(dispDat[i]["Name"])][:]
            fdct = f["%s_fdct"%(dispDat[i]["Name"])][:]

        # Evaluating the CNNs with qdct
        dhsn2_res = evalCNN(qdct, dhsn2_noData)

        # Displaying
        displayRowCNNNoData(
            fdct, qdct, dhsn2_res,
            dispDat[i]["xc"], dispDat[i]["yc"], dispDat[i]["of"],
            dispDat[i]["axs"],
            zoom, vmin, vmax)
        if i==0:
            fs = 20
            dispDat[i]["axs"][0].set_title("QDCT", fontsize= fs)
            dispDat[i]["axs"][1].set_title("DHSN2", fontsize= fs)
            dispDat[i]["axs"][2].set_title("FDCT", fontsize= fs)

    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.show()