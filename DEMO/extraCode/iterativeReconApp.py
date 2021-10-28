import astra
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F


""" Function: norm_l2
    inputs:
        * x: Signal whose l2 norm is computed
    outputs:
        * norm: l2 norm of the signal x
"""
def norm_l2(x):
    norm = np.sum(x**2)
    return(norm)

""" Function: filtCNN
    Description: This function executes noise reduction on
        the signal x with a given cnn. Since this function 
        is meant to be embedded in an iterative reconstruction
        pipeline we need to scale the input x to HU, because
        is what the CNN is trained for. After processing the 
        filtered image is mapped back to attenuation values
        (cm^{-1})
    inputs:
        * x: Signal to be filred with a ggiven cnn
        * cnn: network that performs noise reduction
    outputs:
        * eval_noHU: denoied signal
"""
def filtCNN(x, cnn):
    p = 128 # Padding size before feeding image to CNN
    muw = 0.2 # Assumed water attenuation in cm^{-1}
    
    # The CNNs are trained on Hounsfield Units
    x_hu =  1000*(x - muw)/muw
    
    # Evaluating the image. We use no_grad() to avoid
    # wasting memory on gradients, since we only perform 
    # forward passes
    with torch.no_grad():
        # We pad before feeding the signal
        # to mitigate the effects of zero-padding within
        # the CNN
        inp = torch.from_numpy(
            x_hu).unsqueeze(0).unsqueeze(0).to(torch.float).cuda() 
        inpp = F.pad(inp,(p,p,p,p),"reflect")
        eva = cnn( inpp )
        eva2 = eva.cpu().detach().numpy()[0,0,p:-p,p:-p]
    
    # Returning intensity to attenuation values
    eval_noHU = (muw*eva2/1000 + muw)
    return(eval_noHU)

"""
    forward_fanBeam
    inputs:
        * rec_params: Dictionary with the following reconstruction
            parameters:
            * pix_size
            * no_elems
            * dis_src_iso
            * dis_iso_det
            * im_rows
            * im_cols
            * proj_angs
            * sino
    outputs:
        * proj
"""
def forward_fanBeam(
        image, rec_params):
    
    proj_geom = astra.create_proj_geom(
        'fanflat',
        rec_params["detector"]["pix_size"],
        rec_params["detector"]["no_elems"],
        rec_params["geometry"]["angles"],
        rec_params["geometry"]["dis_src_iso"],
        rec_params["geometry"]["dis_iso_det"])
    
    # Creating volume
    grid_row, grid_col = image.shape
    im_geom = astra.creators.create_vol_geom(grid_row, grid_col )
    proj_id = astra.create_projector('line_fanflat', proj_geom, im_geom)

    sino_id, proj = astra.creators.create_sino(
        data=image, proj_id = proj_id) 

    # Cleaning
    astra.algorithm.delete(sino_id)
    astra.projector.delete(proj_id)
    return(proj)

"""
    backproject_fanBeam
    inputs:
        * rec_params: Dictionary with the following reconstruction
            parameters:
            * pix_size
            * no_elems
            * dis_src_iso
            * dis_iso_det
            * im_rows
            * im_cols
            * proj_angs
            * sino
    outputs:
        * recons
"""
def backproj_fanBeam(sino, im_rows, im_cols, rec_params): 
        
    # Creating projection geometry
    proj_geom = astra.create_proj_geom(
        'fanflat',
        rec_params["detector"]["pix_size"],
        rec_params["detector"]["no_elems"],
        rec_params["geometry"]["angles"],
        rec_params["geometry"]["dis_src_iso"],
        rec_params["geometry"]["dis_iso_det"])
    
    im_geom = astra.creators.create_vol_geom(
        im_rows, im_cols )
    
    proj_id = astra.create_projector('line_fanflat', proj_geom, im_geom)

    
    # Reconstructing
    recons_id, recons = astra.creators.create_backprojection(
       data = sino,
       proj_id = proj_id )

    # Cleaning
    astra.data2d.delete(proj_id)
    astra.data2d.delete(recons_id)
    
    return(recons)

""" Function: reconstruct_fanBeam
    Description: This function performs filtered backprojection
        reconstruction for a fan-beam geometry with a flat panel
        detector
    Inputs:
        * sino: Input sinogram that will be reconstructed
        * im_rows, im_cols: Number of columns and rows of the 
            image that will be outputted by the reconstruction
        * rec_params: Dictionary with the reconstruction parameters
    outputs:
        * recons
"""
def fbp_fanBeam(sino, im_rows, im_cols, rec_params):

    # Creating projection geometry
    proj_geom = astra.create_proj_geom(
        'fanflat',
        rec_params["detector"]["pix_size"],
        rec_params["detector"]["no_elems"],
        rec_params["geometry"]["angles"],
        rec_params["geometry"]["dis_src_iso"],
        rec_params["geometry"]["dis_iso_det"])
    
    # Creating geometry and projector
    im_geom = astra.creators.create_vol_geom(
        im_rows, im_cols )
    proj_id = astra.create_projector(
        'line_fanflat', proj_geom, im_geom)
    
    # Creating sinogram
    sino_id = astra.data2d.create(
        '-sino', proj_geom, sino)
    
    # Reconstructing
    recons_id = astra.data2d.create('-vol', im_geom)
    alg_cfg = astra.astra_dict('FBP_CUDA')
    alg_cfg['ProjectionDataId'] = sino_id
    alg_cfg['ReconstructionDataId'] = recons_id
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data2d.get(recons_id)

    # Cleaning
    astra.data2d.delete(proj_id)
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(recons_id)
    astra.data2d.delete(proj_id)
    
    return(reconstruction)

""" Function: iterative_recons
    Description: This function performs filtered iterative
        reconstruction for a fan-beam geometry with a flat panel
        detector. This is the implementation of Algorithm 1 of the 
        paper
    Inputs:
        * sino: Input sinogram that will be reconstructed
        * im_rows, im_cols: Number of columns and rows of the 
            image that will be outputted by the reconstruction
        * rec_params: Dictionary with the reconstruction parameters
    outputs:
        * recons
"""
def iterative_recons(
    sino, iters, im_rows, im_cols, rec_params, cnn, reg=True, regVal=0.020):
    # Initializing algorithm with 
    # FBP reconstruction
    x_pre = fbp_fanBeam(sino, im_rows, im_cols, rec_params)
    fbp = x_pre
    
    # Iterative reconstruction
    for i in np.arange(iters):
        # Forward projecting and error computation
        Ax = forward_fanBeam(x_pre, rec_params)
        r = Ax - sino  

        # Backprojecting anc computing step-size
        ATr = backproj_fanBeam(r, im_rows, im_cols, rec_params )
        AATr = forward_fanBeam(ATr, rec_params)
        alpha = norm_l2(ATr)/norm_l2(AATr)

        # Updating estimate
        z = x_pre - alpha*ATr
        
        # Applying regularization via
        # projected gradient descent
        if reg:         
            x_pre= z - regVal*(z - filtCNN(z, cnn))
        else:
            x_pre = z
    
    # scaling the resulting images to HU
    muw=0.2
    res = 1000*(x_pre - muw)/muw
    fbp = 1000*(fbp - muw)/muw
    return([ res, fbp ])