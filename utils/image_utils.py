#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
@torch.no_grad()
def psnr_(img1, img2, mask=None):
    img1 = img1.flatten(1)
    img2 = img2.flatten(1)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

    if mask is not None:
        if torch.isinf(psnr).any():
            print('An inf ',mse.mean(),psnr.mean())
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
            psnr = psnr[~torch.isinf(psnr)]
        
    return psnr

@torch.no_grad()
def psnr(img1, img2, mask=None):

    if mask is not None:
        assert mask.shape == img1.shape[-2:], "Mask must match HxW of the image"
        mask = mask.expand_as(img1)
        diff = (img1 - img2) ** 2 * mask
        mse = diff.sum() / mask.sum()
    else:
        mse = ((img1 - img2) ** 2).mean()
    
    mse = torch.clamp(mse, min=1e-10)  # Prevent log(0)
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr_value