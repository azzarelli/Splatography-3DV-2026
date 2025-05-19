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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips



import torch.nn.functional as F

def local_triplet_ranking_loss(depth_pred, depth_target, margin=0.05, patch_size=9):
    """
    Computes triplet ranking loss over local patches in the depth map.
    - depth_pred: predicted depth (Gaussian Splatting) — shape [1, H, W]
    - depth_target: reference depth (Monocular Estimator) — shape [1, H, W]
    """
    B, H, W = 1, *depth_pred.shape[-2:]
    pad = patch_size // 2

    # Unfold to extract local patches
    pred_patches = F.unfold(depth_pred.unsqueeze(0), kernel_size=patch_size, padding=pad)
    target_patches = F.unfold(depth_target.unsqueeze(0), kernel_size=patch_size, padding=pad)

    # pred_patches, target_patches: [1, patch_size*patch_size, H*W]
    pred_center = depth_pred.view(1, 1, -1)  # shape: [1, 1, H*W]
    target_center = depth_target.view(1, 1, -1)

    # Broadcast for anchor-positive comparisons
    pred_diff = pred_patches - pred_center  # shape: [1, P, H*W]
    target_diff = target_patches - target_center

    # Determine sign of relative order in target
    target_order = torch.sign(target_diff)  # -1, 0, 1

    # Use only non-zero relative order (i.e., ignore ties)
    valid = target_order != 0

    # Triplet hinge loss: enforce same sign in prediction as in target
    loss = F.relu(margin - target_order * pred_diff)
    loss = loss[valid].mean()

    return loss

def smooth_boolean_mask(mask, kernel_size=3):
    """
    Returns:
        torch.Tensor: Smoothed boolean mask.
    """
    # Ensure the kernel size is odd
    pad = kernel_size // 2

    # Max Pooling (Dilation) - Expands the mask - mask goes from 0.01 to 1.0
    dilated = F.max_pool2d(((0.99*mask)+0.01).cuda(), kernel_size, stride=1, padding=pad)

    # Min Pooling (Erosion) - Contracts the mask (smoothens edges)
    smoothed_mask = -F.max_pool2d(-dilated, kernel_size, stride=1, padding=pad)

    return smoothed_mask


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, weights=None):
    if weights is None:
        return torch.abs((network_output - gt)).mean()
    else:
        return torch.abs(weights*(network_output - gt)).mean()

def l1_loss_masked(pred, gt, mask=None):
    mask = mask.squeeze(0)
    pred = pred[mask]
    gt = gt[mask]
    return (pred - gt.abs()).mean()

def l1_loss_intense(pred, gt, i):
    diff = (pred - gt).abs()
    return (i*diff).mean() + diff.mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

