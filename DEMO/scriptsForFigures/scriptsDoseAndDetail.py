import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.fftpack import fft2, fftshift
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

""" Function: slice_ssim
    Description: This function computes the MSSIM of an image
    Inputs:
        * sc: input image
        * sc_ref: ground-truth to be compared with
    Outputs:
        * MSSIM value between sc and sc_ref
"""
def slice_ssim(sc, sc_ref):
    mssim, grad, S = ssim(sc, sc_ref, gradient=True, full=True)
    return( mssim )

""" Function: printLineTableSSIM
    Description: This function computes a line of the MSSIM tables
        used in the paper
    Inputs:
        * gt
        * qdct, dhsn2, dhsn1, fbp, red: Output of CNNs being compared
        * letr: Name of the ROI to be displayed
        * x, y: Corners of the squared roi
        * wid: Width in x and y direction of ROI
    Outputs:
        * MSSIM value between sc and sc_ref
"""
def printLineTableSSIM(gt, qdct, dhsn2, dhsn1, fbp, red, letr, x, y, wid):
    vmax=80
    print("%s & $ %1.3f $ & $ %1.3f $ & $ %1.3f $ & $ %1.3f $ & $ %1.3f $ \\\\ \\midrule"%(
        str(letr),
        slice_ssim(gt[x:x+wid,y:y+wid], qdct[x:x+wid,y:y+wid] ),
        slice_ssim(gt[x:x+wid,y:y+wid], dhsn2[x:x+wid,y:y+wid] ),
        slice_ssim(gt[x:x+wid,y:y+wid], dhsn1[x:x+wid,y:y+wid] ),
        slice_ssim(gt[x:x+wid,y:y+wid], fbp[x:x+wid,y:y+wid] ),
        slice_ssim(gt[x:x+wid,y:y+wid], red[x:x+wid,y:y+wid] ),
        )
    )

""" Function: detailPReservationAndDose_image
    Description: This function computes a line of the MSSIM tables
        used in the paper
    Inputs:
        * gt
        * qdct, dhsn2, dhsn1, fbp, red: Output of CNNs being compared
        * letr: Name of the ROI to be displayed
        * x, y: Corners of the squared roi
        * wid: Width in x and y direction of ROI
    Outputs:
        * None
"""
def detailPReservationAndDose_image(dataset_path, whichDose,dhsn1, dhsn2, fbp, red, axs):
    # Window settings
    wl = 50
    ww = 25
    vmax=wl+ww/2
    vmin=wl-ww/2
    
    # Where the patterns start
    ofp = 160
    ofs = 180

    # Location ROIs
    x11 = 17;  y11 = 18
    x12 = 77;  y12 = 18
    x13 = 127; y13 = 18

    x21 = 17;  y21 = 75
    x22 = 77;  y22 = 75
    x23 = 127; y23 = 75

    x31 = 17;  y31 = 122
    x32 = 77;  y32 = 122
    x33 = 127; y33 = 122

    # Other parameters
    wid=40
    lw = 4
    c = "m"
    
    # Reading the ground-truth and noisy images
    with h5py.File(dataset_path, 'r') as f:
        gt =  f["gt"][:]
        noisy = f[whichDose][:]
        s=gt.shape

    # Evaluating scan
    with torch.no_grad():
        p = 128
        inp = torch.from_numpy(noisy).view(1,1,s[0],s[1]).float().cuda()
        inpp = F.pad(inp, (p,p,p,p), "reflect")
        res_dhsn1 = dhsn1(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_dhsn2 = dhsn2(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_fbp = fbp(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_red = red(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()

    
    ## CLOSE UP
    axs[0].imshow(noisy[ofp:-ofs,ofp:-ofs],cmap="gray",vmin=vmin,vmax=vmax,)
    axs[0].yaxis.set_major_locator(plt.NullLocator())
    axs[0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0].set_title("%s input"%(whichDose))

    axs[1].imshow(res_dhsn2[ofp:-ofs,ofp:-ofs],cmap="gray",vmin=vmin,vmax=vmax)
    axs[1].yaxis.set_major_locator(plt.NullLocator())
    axs[1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1].set_title("%s DHSN2"%(whichDose))

    axs[2].imshow(res_dhsn1[ofp:-ofs,ofp:-ofs],cmap="gray",vmin=vmin,vmax=vmax,)
    axs[2].yaxis.set_major_locator(plt.NullLocator())
    axs[2].xaxis.set_major_formatter(plt.NullFormatter())
    axs[2].set_title("%s DHSN1"%(whichDose))

    axs[3].imshow(res_fbp[ofp:-ofs,ofp:-ofs],cmap="gray",vmin=vmin,vmax=vmax,)
    axs[3].yaxis.set_major_locator(plt.NullLocator())
    axs[3].xaxis.set_major_formatter(plt.NullFormatter())
    axs[3].set_title("%s FBPConvNet"%(whichDose))

    axs[4].imshow(res_red[ofp:-ofs,ofp:-ofs],cmap="gray",vmin=vmin,vmax=vmax,)
    axs[4].yaxis.set_major_locator(plt.NullLocator())
    axs[4].xaxis.set_major_formatter(plt.NullFormatter())
    axs[4].set_title("%s RED"%(whichDose))

    # Drawing rectangles around 
    axs[0].add_patch(patches.Rectangle((1,1),510-ofp-ofs,510-ofp-ofs,linewidth=lw,edgecolor=c,facecolor='none'))
    axs[1].add_patch(patches.Rectangle((1,1),510-ofp-ofs,510-ofp-ofs,linewidth=lw,edgecolor=c,facecolor='none'))
    axs[2].add_patch(patches.Rectangle((1,1),510-ofp-ofs,510-ofp-ofs,linewidth=lw,edgecolor=c,facecolor='none'))
    axs[3].add_patch(patches.Rectangle((1,1),510-ofp-ofs,510-ofp-ofs,linewidth=lw,edgecolor=c,facecolor='none'))
    axs[4].add_patch(patches.Rectangle((1,1),510-ofp-ofs,510-ofp-ofs,linewidth=lw,edgecolor=c,facecolor='none'))

""" Function: detailPReservationAndDose_ROIs
    Description: This function displays the ground-truth and ROIs
        for the detail preservation tests
    Inputs:
        * dataset_path: Path to ground-truth image
    Outputs:
        * None
"""
def detailPReservationAndDose_ROIs(dataset_path): 
    # Window settings
    wl = 50
    ww = 25
    vmax=wl+ww/2
    vmin=wl-ww/2
    
    # Where the patterns start
    ofp = 160
    ofs = 180

    # Location ROIs
    x11 = 17;  y11 = 18
    x12 = 77;  y12 = 18
    x13 = 127; y13 = 18

    x21 = 17;  y21 = 75
    x22 = 77;  y22 = 75
    x23 = 127; y23 = 75

    x31 = 17;  y31 = 122
    x32 = 77;  y32 = 122
    x33 = 127; y33 = 122
    
    # Other parameters
    wid=40
    fs = 16
    lw = 4
    c = "m"
    
    # Reading the ground-truth and noisy images
    with h5py.File(dataset_path, 'r') as f:
        gt =  f["gt"][:]
        
    sc=4
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(sc*1.95,sc))
    
    axs[0].imshow(gt,cmap="gray",vmin=vmin,vmax=vmax, interpolation='bilinear')
    axs[0].add_patch(patches.Rectangle((ofp,ofp),ofs,ofs,linewidth=lw,edgecolor='m',facecolor='none'))
    axs[0].yaxis.set_major_locator(plt.NullLocator())
    axs[0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0].set_title("Noiseless ground-truth")

    axs[1].imshow(gt[ofp:-ofs,ofp:-ofs],cmap="gray",vmin=vmin,vmax=vmax, interpolation='bilinear')
    axs[1].add_patch(patches.Rectangle((x11,y11),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x11+wid-7, y11+wid-3, "A",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((x12,y12),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x12+wid-7, y12+wid-3, "B",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((x13,y13),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x13+wid-7, y13+wid-3, "C",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((x21,y21),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x21+wid-7, y21+wid-3, "D",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((x22,y22),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x22+wid-7, y22+wid-3, "E",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((x23,y23),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x23+wid-7, y23+wid-3, "F",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((x31,y31),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x31+wid-7, y31+wid-3, "G",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((x32,y32),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x32+wid-7, y32+wid-3, "H",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((x33,y33),wid,wid,linewidth=2,edgecolor='r',facecolor='none'))
    axs[1].text(x33+wid-7, y33+wid-3, "I",fontsize=fs, color='white',backgroundcolor='red')
    axs[1].add_patch(patches.Rectangle((1,1),510-ofp-ofs,510-ofp-ofs,linewidth=lw,edgecolor=c,facecolor='none'))
    axs[1].yaxis.set_major_locator(plt.NullLocator())
    axs[1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1].set_title("ROIs")
    
    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.show()
    
""" Function: detailPReservation_table
    Description: This function displays a table highlighing the performance
        of the evaluated CNNs to preserve small structures and low-contrast
        content
    Inputs:
        * dataset_path: Path to ground-truth image
        * whichDose: dose level to be loaded
        * dhsn1, dhsn2, fbp, red: CNN models to process the noisy
            image
    Outputs:
        * None
"""  
def detailPReservation_table(dataset_path, whichDose, dhsn2, dhsn1, fbp, red):
    # Where the patterns start
    ofp = 160
    ofs = 180

    # Location ROIs
    x11 = 17;  y11 = 18
    x12 = 77;  y12 = 18
    x13 = 127; y13 = 18

    x21 = 17;  y21 = 75
    x22 = 77;  y22 = 75
    x23 = 127; y23 = 75

    x31 = 17;  y31 = 122
    x32 = 77;  y32 = 122
    x33 = 127; y33 = 122

    # Other parameters
    wid=40
    
    # Reading the ground-truth and noisy images
    with h5py.File(dataset_path, 'r') as f:
        gt =  f["gt"][:]
        noisy = f[whichDose][:]
        s=gt.shape

    # Evaluating scan
    with torch.no_grad():
        p = 128
        inp = torch.from_numpy(noisy).view(1,1,s[0],s[1]).float().cuda()
        inpp = F.pad(inp, (p,p,p,p), "reflect")
        res_dhsn1 = dhsn1(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_dhsn2 = dhsn2(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_fbp = fbp(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_red = red(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
    
    # ROI SSIM
    print("\n\nTable depicting the MSSIM for patterns with different size and contrast")
    print("ROI & %s & DHSN2 & DHSN1 & FBPConvNet & RED \\\\ \\midrule"%(whichDose))
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "A", ofp+x11, ofp+y11, wid)
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "B", ofp+x12, ofp+y12, wid)
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "C", ofp+x13, ofp+y13, wid)
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "D", ofp+x21, ofp+y21, wid)
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "E", ofp+x22, ofp+y22, wid)
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "F", ofp+x23, ofp+y23, wid)
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "G", ofp+x31, ofp+y31, wid)
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "H", ofp+x32, ofp+y32, wid)
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, "I", ofp+x33, ofp+y33, wid)

    
""" Function: doseTableRow
    Description: This function displays a row of a table highlighing the performance
        of the evaluated CNNs to preserve image content with different
        doses
    Inputs:
        * dataset_path: Path to ground-truth image
        * whichDose: dose level to be loaded
        * dhsn1, dhsn2, fbp, red: CNN models to process the noisy
            image
    Outputs:
        * None
"""  
def doseTableRow(dataset_path, whichDose, dhsn2, dhsn1, fbp, red):
    # Where the patterns start
    ofp = 160
    ofs = 180

    # Location ROIs
    x11 = 17;  y11 = 18
    x12 = 77;  y12 = 18
    x13 = 127; y13 = 18

    x21 = 17;  y21 = 75
    x22 = 77;  y22 = 75
    x23 = 127; y23 = 75

    x31 = 17;  y31 = 122
    x32 = 77;  y32 = 122
    x33 = 127; y33 = 122

    # Other parameters
    wid=40
    
    # Reading the ground-truth and noisy images
    with h5py.File(dataset_path, 'r') as f:
        gt =  f["gt"][:]
        noisy = f[whichDose][:]
        s=gt.shape

    # Evaluating scan
    with torch.no_grad():
        p = 128
        inp = torch.from_numpy(noisy).view(1,1,s[0],s[1]).float().cuda()
        inpp = F.pad(inp, (p,p,p,p), "reflect")
        res_dhsn1 = dhsn1(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_dhsn2 = dhsn2(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_fbp = fbp(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
        res_red = red(inpp)[0,0,p:-p,p:-p].cpu().detach().numpy()
    
    printLineTableSSIM(gt, noisy, res_dhsn2, res_dhsn1, res_fbp, res_red, whichDose, ofp, ofp, 512-(ofp+ofs) )

""" Function: tableDose
    Description: This function displays a table highlighing the performance
        of the evaluated CNNs to preserve image content with different
        doses
    Inputs:
        * dataset_path: Path to ground-truth image
        * dhsn1, dhsn2, fbp, red: CNN models to process the noisy
            image
    Outputs:
        * None
"""  
def tableDose(dataset_detail, dhsn2, dhsn1, fbp, red):
    print("\n\nTable depicting the MSSIM for the displayed patch with different dose")
    print("Dose & Noisy & DHSN2 & DHSN1 & FBPConvNet & RED \\\\ \\midrule")
    doseTableRow(dataset_detail, "qdct", dhsn2, dhsn1, fbp, red)
    doseTableRow(dataset_detail, "hdct", dhsn2, dhsn1, fbp, red)
    doseTableRow(dataset_detail, "fdct", dhsn2, dhsn1, fbp, red)
    
""" Function: demoDoseDetailContrast
    Description: This section replicates the images and tables
        displayed in the phantom experiments of dose, and 
        detail preservation
    Inputs:
        * dataset_path: Path to ground-truth image
        * dhsn1, dhsn2, fbp, red: CNN models to process the noisy
            image
    Outputs:
        * None
"""  
def demoDoseDetailContrast(dataset_detail, dhsn2, dhsn1, fbp, red ):
    # Displaying the ground-truth and ROIs used for the 
    # detail preservation section of the measurements
    detailPReservationAndDose_ROIs(dataset_detail)

    # Displaying the processed patches
    sc=3.0
    fig, axs = plt.subplots(nrows=3, ncols=5,figsize=(sc*4.9,sc*3))
    axs_qdct = [axs[0,0], axs[0,1], axs[0,2], axs[0,3], axs[0,4]]
    detailPReservationAndDose_image(dataset_detail, "qdct", dhsn2, dhsn1, fbp, red, axs_qdct)
    axs_hdct = [axs[1,0], axs[1,1], axs[1,2], axs[1,3], axs[1,4]]
    detailPReservationAndDose_image(dataset_detail, "hdct", dhsn2, dhsn1, fbp, red, axs_hdct)
    axs_fdct = [axs[2,0], axs[2,1], axs[2,2], axs[2,3], axs[2,4]]
    detailPReservationAndDose_image(dataset_detail, "fdct", dhsn2, dhsn1, fbp, red, axs_fdct)
    plt.subplots_adjust(wspace=-0.4, hspace= 0.2)
    plt.show()

    # Showing the table with the measurements of the ROIs on the QDCT
    detailPReservation_table(dataset_detail, "qdct",dhsn2, dhsn1, fbp, red)


    # Showing the table with the measurements of the ROIs on the QDCT
    tableDose(dataset_detail, dhsn2, dhsn1, fbp, red)