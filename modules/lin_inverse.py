#!/usr/bin/env python
'''
Using kornia here for Radon transform because it allows for differentiable CV operations
'''
import torch
import kornia

def radon(imten, angles, is_3d=False):
    '''
        Compute forward radon operation
        
        Inputs:
            imten: (1, nimg, H, W) image tensor
            angles: (nangles) angles tensor -- should be on same device as 
                imten
        Outputs:
            sinogram: (nimg, nangles, W) sinogram
    '''
    nangles = len(angles)
    imten_rep = torch.repeat_interleave(imten, nangles, 0)
    
    imten_rot = kornia.geometry.rotate(imten_rep, angles)
    
    if is_3d:
        sinogram = imten_rot.sum(2).squeeze().permute(1, 0, 2)
    else:
        sinogram = imten_rot.sum(2).squeeze()
        
    return sinogram

