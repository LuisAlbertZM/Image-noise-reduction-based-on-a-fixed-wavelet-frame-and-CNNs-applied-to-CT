import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.fftpack import fft2, fftshift

""" Function: radial_profile
    Description: This function generates a radial average of a given image
    Inputs:
        * data: Input image in which the radial average will eb computed
        * center: Center in which the average is computed
    Outputs:
        * radialprofile: Radial profile
"""
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


""" Function: compute_NPS_images
    Description: This function displays
    Inputs:
        * dataRoot: Path to dataset
        * dataset: Name of the .h5 file containing noise realizations to compute NPS
        * cnn: Pytorch model to be tested
        * which_cnn: Name of the CNN to be used for display
        * pltColor: Color of the line in the radial profiles to be used for the CNN
        * axs: Axes to be used in the subplot
        * qdctOnly: If True the CNN is ignored and the NPS of the unprocessed noise
            is computed
    Outputs:
        * None.
"""
def compute_NPS_images(
    dataNPS_0, dataNPS_1, dataNPS_2, dataNPS_3, dataNPS_4,
    cnn, which_cnn, pltColor, fig, axs, qdctOnly=False):
    
    Ns = 50
    qdct = np.zeros((512,50,512)) 
    
    # We have 50 noise reealizations to compute the NPS
    with h5py.File(dataNPS_0, 'r') as f:
        gt =  f["gt"][:]
        qdct[:,0:10,:] = f["qdct"][:]   
    with h5py.File(dataNPS_1, 'r') as f:
        qdct[:,10:20,:] = f["qdct"][:]
    with h5py.File(dataNPS_2, 'r') as f:
        qdct[:,20:30,:] = f["qdct"][:]
    with h5py.File(dataNPS_3, 'r') as f:
        qdct[:,30:40,:] = f["qdct"][:]
    with h5py.File(dataNPS_4, 'r') as f:
        qdct[:,40:50,:] = f["qdct"][:]
    s=gt.shape

    # Computing estimate of NPS
    var_cnn = np.zeros(s)
    var_qdct = np.zeros(s)
    nps_cnn = np.zeros(s)
    nps_qdct = np.zeros(s)
    for i in np.arange(Ns):
        # Evaluating scan
        with torch.no_grad():
            ofs = 128
            if not qdctOnly:
                inp = torch.from_numpy(qdct[:,i,:]).view(1,1,s[0],s[1]).float().cuda()
                inpp = F.pad(inp,(ofs,ofs,ofs,ofs),"reflect")
                res = cnn( inpp  )[0,0,ofs:-ofs,ofs:-ofs].cpu().detach().numpy()
                var_cnn += (res - qdct[:,i,:])**2/Ns
                nps_cnn += np.absolute(fftshift(fft2(res-gt)))**2/(Ns*s[0]*s[1])
            var_qdct += (qdct[:,i,:] - gt)**2/Ns
            nps_qdct += np.absolute(fftshift(fft2(qdct[:,i,:]-gt)))**2/(Ns*s[0]*s[1])

    # Radial profile of NPS
    if not qdctOnly:
        rp_cnn = radial_profile(nps_cnn,(255.5,255.5))
    rp_qdct = radial_profile(nps_qdct,(255.5,255.5))
    xval = np.sqrt(2)*0.5*np.arange(rp_qdct.shape[0])/rp_qdct.shape[0]
    
    # Display pixel variance
    if not qdctOnly:
        img = axs[0].imshow(var_cnn, cmap = "hot", vmin = 0, vmax=100)
    else:
        img = axs[0].imshow(var_qdct, cmap = "hot", vmin = 0, vmax=100)
    fig.colorbar(img, orientation='vertical',ax=axs[0])
    axs[0].yaxis.set_major_locator(plt.NullLocator())
    axs[0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0].set_title("Est. Pix. var. %s"%(which_cnn) )
    
    # Display NPS
    if not qdctOnly:
        img = axs[1].imshow(nps_cnn, cmap = "hot", vmin = 0, vmax=100)
    else:
        img = axs[1].imshow(nps_qdct, cmap = "hot", vmin = 0, vmax=100)
    fig.colorbar(img, orientation='vertical',ax=axs[1])
    axs[1].yaxis.set_major_locator(plt.NullLocator())
    axs[1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1].set_title("NPS")
    
    # Display radial profile of NPS
    axs[2].plot(xval, rp_qdct, "g", label="QDCT")
    if not qdctOnly:
        axs[2].plot(xval, rp_cnn, pltColor, label=which_cnn)
    axs[2].legend()
    axs[2].set_xlim([0, 0.5])
    axs[2].set_ylim([0, 100])
    axs[2].set_title("Radial profile NPS")
    axs[2].set_ylabel('Noise power [HU]$^2$', fontsize=12)
    axs[2].set_xlabel('Relative frequency [fs]', fontsize=12)

""" Function: compute_NPS_images
    Description: This function displays
    Inputs:
        * dataNPS: Path to .h5 file containing noise realizations to generate NPS
        * dhsn1, dhsn2, fbp, red: Pytorch model to be tested
    Outputs:
        * None.
"""
def npsOfTestedCNNs(dataNPS_0, dataNPS_1, dataNPS_2, dataNPS_3, dataNPS_4,
                     dhsn1, dhsn2, fbp, red):
    # Generating figure
    fig, axs = plt.subplots(nrows=5, ncols=3,figsize=(8,16.0))
    
    # Each row of the figure is a different CNN
    compute_NPS_images(
        dataNPS_0, dataNPS_1, dataNPS_2, dataNPS_3, dataNPS_4, 
        "   ", "     ", " ", fig, [axs[0,0], axs[0,1], axs[0,2]], True)
    compute_NPS_images(
        dataNPS_0, dataNPS_1, dataNPS_2, dataNPS_3, dataNPS_4, 
        dhsn1, "DHSN1", "m", fig, [axs[1,0], axs[1,1], axs[1,2]])
    compute_NPS_images(
        dataNPS_0, dataNPS_1, dataNPS_2, dataNPS_3, dataNPS_4, 
        dhsn2, "DHSN2", "b", fig, [axs[2,0], axs[2,1], axs[2,2]])
    compute_NPS_images(
        dataNPS_0, dataNPS_1, dataNPS_2, dataNPS_3, dataNPS_4, 
        fbp,   "FBP",   "c", fig, [axs[3,0], axs[3,1], axs[3,2]])
    compute_NPS_images(
        dataNPS_0, dataNPS_1, dataNPS_2, dataNPS_3, dataNPS_4, 
        red,   "RED",   "r", fig, [axs[4,0], axs[4,1], axs[4,2]])
    
    # Displaying
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=1.2, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    plt.show()