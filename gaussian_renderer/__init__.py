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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
import torch.nn.functional as F
# from gaussian_renderer.pytorch_render import GaussRenderer

from gsplat.rendering import rasterization

# RENDER = GaussRenderer()

def quaternion_rotate(q, v):
    q_vec = q[:, :3]
    q_w = q[:, 3].unsqueeze(1)
    t = 2 * torch.cross(q_vec, v, dim=1)
    return v + q_w * t + torch.cross(q_vec, t, dim=1)

def rotated_softmin_axis_direction(r, s, temperature=10.0):
    # s: (N, 3), we want the direction of the smallest abs scale
    abs_s = torch.abs(s)

    # Step 1: Compute softmin weights (lower abs(s) => higher weight)
    weights = F.softmax(-abs_s * temperature, dim=1)  # (N, 3)

    # Step 2: Basis axes: x, y, z
    basis = torch.eye(3, device=s.device).unsqueeze(0)  # (1, 3, 3)

    # Step 3: Weighted sum of basis vectors
    soft_axis = torch.bmm(weights.unsqueeze(1), basis.repeat(s.size(0), 1, 1)).squeeze(1)  # (N, 3)

    # Step 4: Rotate the direction
    rotated = quaternion_rotate(r, soft_axis)  # (N, 3)

    return rotated


def differentiable_cdf_match(source, target, eps=1e-5):
    """
    Remap 'source' values to match the CDF of 'target', using differentiable quantile mapping.
    Works in older PyTorch versions (no torch.interp).
    """
    # Sort source and target
    source_sorted, _ = torch.sort(source)
    target_sorted, _ = torch.sort(target)

    # Build uniform CDF positions
    cdf_vals = torch.linspace(0.0, 1.0, len(source_sorted), device=source.device)

    # Step 1: Get CDF values of 'source' values via inverse CDF
    # Interpolate where each source value would sit in its own sorted list
    idx = torch.searchsorted(source_sorted, source, right=False).clamp(max=len(cdf_vals) - 2)
    x0 = source_sorted[idx]
    x1 = source_sorted[idx + 1]
    y0 = cdf_vals[idx]
    y1 = cdf_vals[idx + 1]
    t = (source - x0) / (x1 - x0 + eps)
    source_cdf = y0 + t * (y1 - y0)

    # Step 2: Map CDF to target values (i.e., inverse CDF of target)
    idx = torch.searchsorted(cdf_vals, source_cdf, right=False).clamp(max=len(target_sorted) - 2)
    x0 = cdf_vals[idx]
    x1 = cdf_vals[idx + 1]
    y0 = target_sorted[idx]
    y1 = target_sorted[idx + 1]
    t = (source_cdf - x0) / (x1 - x0 + eps)
    matched = y0 + t * (y1 - y0)

    return matched
        
from torch_cluster import knn
def compute_alpha_interval(A, B, cov6A, cov6B, alpha_threshold=0.1):
    """
    A is the target point
    B is the outer point
    """
    # Step 1: Unpack 6D compact covariances into full 3x3 matrices
    N = cov6A.shape[0]
    device = cov6A.device
    covA = torch.zeros((N, 3, 3), device=cov6A.device, dtype=cov6A.dtype)
    covA[:, 0, 0] = cov6A[:, 0]
    covA[:, 1, 1] = cov6A[:, 1]
    covA[:, 2, 2] = cov6A[:, 2]
    covA[:, 0, 1] = covA[:, 1, 0] = cov6A[:, 3]
    covA[:, 0, 2] = covA[:, 2, 0] = cov6A[:, 4]
    covA[:, 1, 2] = covA[:, 2, 1] = cov6A[:, 5]
    N = cov6B.shape[0]
    covB = torch.zeros((N, 3, 3), device=cov6A.device, dtype=cov6A.dtype)
    covB[:, 0, 0] = cov6B[:, 0]
    covB[:, 1, 1] = cov6B[:, 1]
    covB[:, 2, 2] = cov6B[:, 2]
    covB[:, 0, 1] = covB[:, 1, 0] = cov6B[:, 3]
    covB[:, 0, 2] = covB[:, 2, 0] = cov6B[:, 4]
    covB[:, 1, 2] = covB[:, 2, 1] = cov6B[:, 5]
    # Add small regularization for stability
    # eps = 1e-6
    # cov[:, range(3), range(3)] += eps
    covA_inv = torch.linalg.inv(covA)
    covB_inv = torch.linalg.inv(covB)
    
    # Step 2: Compute λ = d^T @ Σ⁻¹ @ d (Mahalanobis squared along direction d)
    d_AB = torch.nn.functional.normalize(A-B, dim=1).unsqueeze(1)  #.unsqueeze(1)  # (N, 1, 3)
    λB = torch.bmm(torch.bmm(d_AB, covB_inv), d_AB.transpose(1, 2)).squeeze(-1).squeeze(-1).clamp(min=1e-10)  # (N,)    
    c = -2.0 * torch.log(torch.tensor(0.05, device=device, dtype=torch.float)) # LHS of alpha distance function
    t = torch.sqrt(c / λB).clamp(min=1e-10).unsqueeze(-1) # (N,) the distance along A-B where t is 0.01 - we want to select the positive one (by default)
    B_ = B + t*d_AB.squeeze(1)
    
    
    # For the ray B-A the t w.r.t A is 1-t (for the first intersection along the ray) and 1 + t (for the second)
    d_BA = torch.nn.functional.normalize(B_- A, dim=1)  #.unsqueeze(1)  # (N, 1, 3)
    
    λA = torch.bmm(torch.bmm(d_BA.unsqueeze(1), covA_inv), d_BA.unsqueeze(1).transpose(1, 2)).squeeze(-1).squeeze(-1).clamp(min=1e-10).unsqueeze(-1)  # (N,)
    alpha = torch.exp(-0.5* λA) # * t_BA.pow(2))
    return alpha

def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           stage="fine", view_args=None, G=None, kernel_size=0.1):
    """
    Render the scene.
    """

    extras = None

    means3D_ = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D_.device).repeat(means3D_.shape[0], 1)
    # colors = pc.get_color
    colors = pc.get_features

    opacity = pc.get_opacity.clone().detach()

    scales = pc.get_scaling_with_3D_filter
    rotations = pc._rotation
    target_mask = pc.target_mask

    if view_args is not None:

        
        if view_args['finecoarse_flag']:
            means3D, rotations, opacity, colors, extras = pc._deformation(
                point=means3D_, 
                rotations=rotations,
                scales=scales,
                times_sel=time, 
                h_emb=opacity,
                shs=colors,
                view_dir=viewpoint_camera.direction_normal(),
                target_mask=target_mask
            )
        else:
            means3D, extras = means3D_, None
            opacity = pc.get_coarse_opacity_with_3D_filter

        
            
    else:
        means3D, rotations, opacity, colors, extras = pc._deformation(
            point=means3D_, 
            rotations=rotations,
            scales=scales,
            times_sel=time, 
            h_emb=opacity,
            shs=colors,
            view_dir=viewpoint_camera.direction_normal(),
            target_mask=target_mask
        )
    
    opacity = pc.get_fine_opacity_with_3D_filter(opacity)
    rotation = pc.rotation_activation(rotations)

    show_mask = 0
    if view_args is not None and stage != 'test':
        if view_args['viewer_status']:
            show_mask = view_args['show_mask'] 
            
            mask = ((pc.get_wopac**2 *2000.) > view_args['w_thresh'])
            mask = torch.logical_and(mask, (pc.get_hopac > view_args['h_thresh']).squeeze(-1))

            mask = mask.squeeze(-1)
            if show_mask == 1:
                mask = torch.logical_and(target_mask, mask)
            elif show_mask == -1:
                mask = torch.logical_and(~target_mask, mask)
            
            if mask is not None:
                means3D = means3D[mask]
                colors = colors[mask]
                opacity = opacity[mask]
                scales = scales[mask]
                rotation = rotation[mask]
                
            if view_args['full_opac']:
                opacity = torch.ones_like(opacity).cuda()
                colors = (means3D - means3D_).abs()
    else:
        view_args= {'vis_mode':'render'}

            
    # print(.shape, means3D.shape)
    rendered_image, rendered_depth, norms = None, None, None
    if stage == 'test-foreground':
        # distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
        # mask = distances > 0.3
        mask = target_mask #torch.logical_and(, mask)

        means3D = means3D[mask]
        rotation = rotation[mask]
        scales = scales[mask]
        opacity = opacity[mask]
        colors = colors[mask]

        rendered_image, alpha, _ = rasterization(
                        means3D, rotation, scales, opacity.squeeze(-1), colors,

            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
        extras = alpha.squeeze(0).permute(2,0,1)
    elif stage == 'test-full':
        distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
        mask = distances < 2.
        mask = torch.logical_and(~target_mask, mask)
        mask = ~mask
        means3D = means3D[mask]
        rotation = rotation[mask]
        scales = scales[mask]
        opacity = opacity[mask]
        colors = colors[mask]

        rendered_image, alpha, _ = rasterization(
                        means3D, rotation, scales, opacity.squeeze(-1), colors,

            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
        
    elif view_args['vis_mode'] in ['render']:
        distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
        mask = distances > 0.3
        means3D = means3D[mask]
        rotation = rotation[mask]
        scales = scales[mask]
        opacity = opacity[mask]
        colors = colors[mask]

        rendered_image, alpha, _ = rasterization(
                        means3D, rotation, scales, opacity.squeeze(-1), colors,

            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
        
    elif view_args['vis_mode'] == 'alpha':
        _, rendered_image, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='RGB',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
    elif view_args['vis_mode'] == 'D':
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='D',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
        
    elif view_args['vis_mode'] == 'ED':
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1), colors,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='ED',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
    elif view_args['vis_mode'] == 'norms':
        # rendered_image = linear_rgb_to_srgb(rendered_image)
        norms = rotated_softmin_axis_direction(rotation, scales)

        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),norms,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)

    elif view_args['vis_mode'] == 'xyz':
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),means3D,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    elif view_args['vis_mode'] == 'dxyz_1':
        residual = torch.norm(means3D-means3D_, dim=-1).unsqueeze(-1).repeat(1,3)
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),residual,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    elif view_args['vis_mode'] == 'dxyz_3':
        residual = (means3D-means3D_).abs()
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),residual,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    elif view_args['vis_mode'] == 'extra':
        # get the nearest xyz neighbours 
        x = means3D
        K=3
        edge_index = knn(x, x, k=K)
        row, col = edge_index
        # 1. Get features of the neighbors
        neighbors = x[col]  # shape: (N*K, 3)
        # 2. Build index matrix to reshape into (N, K, 3)
        sorted_row, perm = row.sort(stable=True)
        sorted_col = col[perm]
        neighbors = x[sorted_col]  # (N*K, 3), sorted by query index
        neighbors = neighbors.view(means3D.shape[0],K ,3)
        cov6A = pc.covariance_activation(scales.detach(), 1, rotation.detach())
        cov6B = cov6A[sorted_col].view(means3D.shape[0],K ,6)
        
        # colors = colors[sorted_col].view(means3D.shape[0],K ,3)[:,1, :]

        for k in range(K):
            if k == 0:
                alpha = compute_alpha_interval(x, neighbors[:, 1, :], cov6A, cov6B[:, 1, :]) #.repeat(1,3) #.unsqueeze(-1)
            else:
                alpha += compute_alpha_interval(x, neighbors[:, 1, :], cov6A, cov6B[:, 1, :]) #.repeat(1,3) #.unsqueeze(-1)

        alpha = alpha / K
        rendered_image, alpha, _ = rasterization(
            means3D, rotation, scales, alpha.squeeze(-1), colors,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
        
    return {
        "render": rendered_image,
        "extras":extras # A dict containing mor point info
        # 'norms':rendered_norm, 'alpha':rendered_alpha
        }


def srgb_to_linear_rgb(srgb: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB to linear RGB.
    Expects input in [0, 1] range. Works on CUDA tensors.
    """
    threshold = 0.04045
    below = srgb <= threshold
    above = ~below
    linear = torch.empty_like(srgb)
    linear[below] = srgb[below] / 12.92
    linear[above] = ((srgb[above] + 0.055) / 1.055) ** 2.4
    return linear


def linear_rgb_to_srgb(linear_rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert linear RGB to sRGB.
    Expects input in [0, 1] range. Works on CUDA tensors.
    """
    threshold = 0.0031308
    below = linear_rgb <= threshold
    above = ~below
    srgb = torch.empty_like(linear_rgb)
    srgb[below] = linear_rgb[below] * 12.92
    srgb[above] = 1.055 * (linear_rgb[above] ** (1/2.4)) - 0.055
    return srgb

def get_edges(mask):
    # Assume mask is float32 (0.0 or 1.0), shape (H, W)
    mask = mask.unsqueeze(0).float()  # (1, 1, H, W)

    laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                       [1, -4, 1],
                                       [0, 1, 0]]]], dtype=mask.dtype, device=mask.device)

    edge = F.conv2d(mask, laplacian_kernel, padding=1).abs()
    edge = edge.squeeze(0).squeeze(0)  # back to (H, W)

    mask_ =  (edge > 0).float()
    mask_ = mask_.unsqueeze(0).unsqueeze(0)
    
    kernel_size=3
    dilated = F.max_pool2d(mask_, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    edge_region =  dilated.squeeze(0).squeeze(0)

    surrounding = (edge_region - mask_).clamp(0, 1)

    # Interior mask
    interior = (mask - mask_).clamp(0, 1)
    return interior.squeeze(0).squeeze(0)

from utils.loss_utils import l1_loss, l1_loss_masked,local_triplet_ranking_loss
def render_coarse_batch_vanilla(
    viewpoint_cams, pc):
    """
    Render the scene.
    """
    means3D = pc.get_xyz    
    scales = pc.get_scaling_with_3D_filter
    rotations = pc.rotation_activation(pc._rotation)
    # colors = pc.get_color
    colors = pc.get_features
    opacity = pc.get_fine_opacity_with_3D_filter(pc.get_hopac)

    means3D = means3D[~pc.target_mask]
    rotations = rotations[~pc.target_mask]
    scales = scales[~pc.target_mask]
    colors = colors[~pc.target_mask]
    opacity = opacity[~pc.target_mask]
    
    L1 = 0.
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
        mask = distances > 0.3
        
        means3D_final = means3D[mask]
        rotations_final = rotations[mask]
        scales_final = scales[mask]
        colors_final = colors[mask]
        opacity_final = opacity[mask]

        background = torch.rand(1, 3).cuda() 

        rgb, _, _ = rasterization(
            means3D_final, rotations_final, scales_final, 
            opacity_final.squeeze(-1),colors_final,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree,
            backgrounds=background
        )
        rgb = rgb.squeeze(0).permute(2,0,1)
        
        # Train the backgroudn
        gt_img = viewpoint_camera.original_image.cuda()
        mask = viewpoint_camera.mask.cuda() > 0. # invert binary mask
        inv_mask = 1. - mask.float() 
        gt = gt_img * inv_mask + (mask)*background.permute(1,0).unsqueeze(-1)

        L1 += l1_loss(rgb, gt)
    
    return  L1

def render_coarse_batch(
    viewpoint_cams, pc, pipe, bg_color: torch.Tensor,scaling_modifier=1.0,
    stage="fine", iteration=0,kernel_size=0.1):
    """
    Render the scene.
    """
    means3D = pc.get_xyz    
    scales = pc.get_scaling_with_3D_filter
    rotations = pc.rotation_activation(pc._rotation)
    # colors = pc.get_color
    colors = pc.get_features
    opacity = pc.get_fine_opacity_with_3D_filter(pc.get_hopac)

    means3D = means3D[~pc.target_mask]
    rotations = rotations[~pc.target_mask]
    scales = scales[~pc.target_mask]
    colors = colors[~pc.target_mask]
    opacity = opacity[~pc.target_mask]
    
    L1 = 0.
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
        mask = distances > 0.3
        
        means3D_final = means3D[mask]
        rotations_final = rotations[mask]
        scales_final = scales[mask]
        colors_final = colors[mask]
        opacity_final = opacity[mask]

        rgb, _, _ = rasterization(
            means3D_final, rotations_final, scales_final, 
            opacity_final.squeeze(-1),colors_final,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rgb = rgb.squeeze(0).permute(2,0,1)
        
        # Train the backgroudn
        gt_img = viewpoint_camera.original_image.cuda()
        mask = viewpoint_camera.mask.cuda() > 0. # invert binary mask
        inv_mask = 1. - mask.float() 
        gt = gt_img * inv_mask
        # Blue gt and fill mask regions with this
        kernel_size = 51
        kernel = torch.ones((3, 1, kernel_size, kernel_size), dtype=gt.dtype, device=gt.device)
        kernel /= kernel_size * kernel_size

        # Apply depthwise convolution (groups=3 for 3 channels)
        blurred = F.conv2d(gt.unsqueeze(0) , kernel, padding=kernel_size//2, groups=3).squeeze(0)
        mask = mask.unsqueeze(0).repeat(3,1,1)
        gt[mask] = blurred[mask]

        L1 += l1_loss(rgb, gt)
    
    return  L1

def render_coarse_batch_target(viewpoint_cams, pc, pipe, bg_color: torch.Tensor,scaling_modifier=1.0,
    stage="fine", iteration=0,kernel_size=0.1):
    """
    Render the scene.
    """
    # means3D = pc.get_xyz    
    # scales = pc.get_scaling_with_3D_filter
    # rotations = pc._rotation
    # # colors = pc.get_color
    # colors = pc.get_features

    means3D = pc.get_xyz    
    scales = pc.get_scaling_with_3D_filter
    rotations = pc._rotation
    colors = pc.get_features

    means3D = means3D[pc.target_mask]
    rotations = rotations[pc.target_mask]
    scales = scales[pc.target_mask]
    colors = colors[pc.target_mask]
    
    L1 = 0.
    time = torch.tensor(viewpoint_cams[0].time).to(means3D.device).repeat(means3D.shape[0], 1).detach()
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        time = time*0. + viewpoint_camera.time

        means3D_final, rotations_final, opacity_final, colors_final, norms = pc._deformation(
            point=means3D, 
            rotations=rotations,
            scales = scales,
            times_sel=time, 
            h_emb=pc.get_opacity[pc.target_mask],
            shs=colors,
            view_dir=viewpoint_camera.direction_normal(),
            target_mask=None,
        )
        
        opacity_final = pc.get_fine_opacity_with_3D_filter(opacity_final)
        rotations_final = pc.rotation_activation(rotations_final)
        # As we take the NN from some random time step, lets re-calc it frequently
        if (iteration % 500 == 0 and idx == 0) or pc.target_neighbours is None:
            pc.update_neighbours(means3D_final)
        background = torch.rand(1, 3).cuda() 
        rgb, _, _ = rasterization(
            means3D_final, rotations_final, scales, 
            opacity_final.squeeze(-1), colors_final,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree,
            backgrounds=background
        )
        
        rgb = rgb.squeeze(0).permute(2,0,1)
                
        gt = viewpoint_camera.original_image.cuda()
        mask = viewpoint_camera.mask.cuda() # > 0. # invert binary mask
        gt = gt*mask + (1.-mask)*background.permute(1,0).unsqueeze(-1)
        
        L1 += l1_loss(rgb, gt)
    return  L1



def render_batch(
    viewpoint_cams, pc, datasettype):
    """
    Render the scene.
    """
    means3D = pc.get_xyz
    scales = pc.get_scaling_with_3D_filter
    rotations = pc._rotation
    # colors = pc.get_color
    colors = pc.get_features

    opacity = pc.get_opacity
    
    L1 = 0.

    time = torch.tensor(viewpoint_cams[0].time).to(means3D.device).repeat(means3D.shape[0], 1).detach()
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        time = time*0. +viewpoint_camera.time
        
        
        means3D_final, rotations_final, opacity_final, colors_final, norms = pc._deformation(
            point=means3D, 
            rotations=rotations,
            scales = scales,
            times_sel=time, 
            h_emb=opacity,
            shs=colors,
            view_dir=viewpoint_camera.direction_normal(),
            target_mask=pc.target_mask,
        )
                
        opacity_final = pc.get_fine_opacity_with_3D_filter(opacity_final)        
        rotations_final = pc.rotation_activation(rotations_final)
        
        # For vivo
        if datasettype == 'condense':
            distances = torch.norm(means3D_final - viewpoint_camera.camera_center.cuda(), dim=1)
            mask = distances > 0.3
            means3D_final = means3D_final[mask]
            rotations_final = rotations_final[mask]
            scales_final = scales[mask]
            opacity_final = opacity_final[mask]
            colors_final = colors_final[mask]
        else:
            scales_final = scales
        
        # Set up rasterization configuration
        rgb, alpha, _ = rasterization(
            means3D_final, rotations_final, scales_final, 
            opacity_final.squeeze(-1), colors_final,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        rgb = rgb.squeeze(0).permute(2,0,1)
        gt_img = viewpoint_camera.original_image.cuda()

        # For ViVo
        if datasettype == 'condense': # remove black edge from loss (edge from undistorting the images)
            L1 += l1_loss(rgb[:, 100:-100, 100:-100], gt_img[:, 100:-100, 100:-100])
        else:
            L1 += l1_loss(rgb, gt_img)
   
    return L1


def render_depth_batch(
    viewpoint_cams, canon_cams,
    pc
    ):
    """
    Render the scene.
    """
    means3D = pc.get_xyz.detach()
    scales = pc.get_scaling_with_3D_filter.detach()
    rotations = pc._rotation.detach()
    colors = pc.get_features.detach()
    opacity = pc.get_opacity.detach()
    
    L1 = 0.

    time = torch.tensor(viewpoint_cams[0].time).to(means3D.device).repeat(means3D.shape[0], 1).detach()
    for viewpoint_camera, canon_camera in zip(viewpoint_cams, canon_cams):
        time = time*0. +viewpoint_camera.time
        
        # Render canon depth
        with torch.no_grad():
            distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
            mask = distances > 0.3

            means3D_final = means3D[mask]
            rotations_final = rotations[mask]
            scales_final = scales[mask]
            opacity_final = pc.get_coarse_opacity_with_3D_filter[mask].detach()
            colors_final = colors[mask]
            
            D, _, _ = rasterization(
                means3D_final, rotations_final, scales_final, 
                opacity_final.squeeze(-1),colors_final,
                viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
                viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
                viewpoint_camera.image_width, 
                viewpoint_camera.image_height,
                
                render_mode='D',
                rasterize_mode='antialiased',
                eps2d=0.1,
                sh_degree=pc.active_sh_degree
            )
            D = D.squeeze(0).permute(2,0,1)

        # Deform for current time step
        means3D_final, rotations_final, opacity_final, colors_final, norms = pc._deformation(
            point=means3D, 
            rotations=rotations,
            scales = scales,
            times_sel=time, 
            h_emb=opacity,
            shs=colors,
            view_dir=viewpoint_camera.direction_normal(),
            target_mask=pc.target_mask,
        )
        opacity_final = pc.get_fine_opacity_with_3D_filter(opacity_final)        
        rotations_final = pc.rotation_activation(rotations_final)
        
        # Filter near-camera 3D viewpointss
        distances = torch.norm(means3D_final - viewpoint_camera.camera_center.cuda(), dim=1)
        mask = distances > 0.3
        means3D_final = means3D_final[mask]
        rotations_final = rotations_final[mask]
        scales_final = scales[mask]
        opacity_final = opacity_final[mask]
        colors_final = colors_final[mask]

        # Set up rasterization configuration
        D_t, _, _ = rasterization(
            means3D_final, rotations_final.detach(), scales_final.detach(), 
            opacity_final.squeeze(-1).detach(),colors_final.detach(),
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            render_mode='D',
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.active_sh_degree
        )
        
        D_t = D_t.squeeze(0).permute(2,0,1)
        Q = (D-D_t).abs()

        Q = (Q - Q.min())/ (Q.max() - Q.min())
        Q_inv = 1. - Q
        with torch.no_grad():
            I_t = viewpoint_camera.original_image.cuda()
            I = canon_camera.original_image.cuda()
            P = (I-I_t).abs()
            P = (P - P.min())/(P.max() - P.min())
        
        L1 += (P*Q_inv).mean()
            
    
    return L1

def render_motion_point_mask(pc):
    """
    Render the scene.
    """
    means3D = pc.get_xyz.detach()
    scales = pc.get_scaling_with_3D_filter.detach()
    rotations = pc._rotation.detach()
    colors = pc.get_features.detach()
    opacity = pc.get_opacity.detach()
    
    L1 = 0.

    time = torch.zeros_like(means3D[:, 0], device=means3D.device).unsqueeze(-1)
    means3D_collection = []
    for i in range(10):
        time = time*0. + float(i)*0.1

        means3D_final, rotations_final, opacity_final, colors_final, norms = pc._deformation(
            point=means3D, 
            rotations=rotations,
            scales = scales,
            times_sel=time, 
            h_emb=opacity,
            shs=colors,
            view_dir=None,
            target_mask=pc.target_mask,
        )
        
        means3D_collection.append(means3D_final[pc.target_mask].unsqueeze(0))
    
    means3D_collection = torch.cat(means3D_collection, dim=0) # K, N, 3, where K=10 for each time step
    displacement = ((means3D_collection - means3D_collection.mean(dim=0))**2).sum(dim=2).sqrt()  # K, N, 3
    motion_metric = displacement.mean(dim=0) # shape (N,)

    threshold = torch.quantile(motion_metric, 0.9)

    mask = (motion_metric >= threshold)
    
    final_mask = torch.zeros_like(pc.target_mask, device=means3D.device)
    final_mask[pc.target_mask] = mask
    return final_mask

def make_T(p):
    """4x4 translation by +p (world space)."""
    T = torch.eye(4, device=p.device, dtype=p.dtype)
    T[:3, 3] = p
    return T

def make_Rx(theta, device, dtype):
    """4x4 rotation about world X axis by theta (right-handed)."""
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.eye(4, device=device, dtype=dtype)
    R[1,1] =  c; R[1,2] = -s
    R[2,1] =  s; R[2,2] =  c
    return R

def rotate_view(viewmat, viewpoint_camera, cent, offset=0):
    T = (int(viewpoint_camera.time * 1800) % 900)+offset
    th = torch.linspace(0, 2 * math.pi, 900)[T]
    
    # --- build orbit transform in world space about 'cent' ---
    cx, cy, cz = cent[0]                           # [3]
    T_neg = make_T(-torch.tensor([cx, cy, cz], device=viewmat.device, dtype=viewmat.dtype))
    T_pos = make_T( torch.tensor([cx, cy, cz], device=viewmat.device, dtype=viewmat.dtype))
    R_x  = make_Rx(th, viewmat.device, viewmat.dtype)

    # Current world->camera and its inverse (camera pose in world)
    W2C = viewmat[0]                               # [4,4]
    C2W = torch.inverse(W2C)                       # [4,4]

    # Rotate the *camera pose* around world X axis about 'cent'
    # cam_to_world' = T_pos * R_x * T_neg * cam_to_world
    C2W_new = T_pos @ R_x @ T_neg @ C2W

    # Back to world->camera for the rasterizer
    return torch.inverse(C2W_new).unsqueeze(0)  # [1,4,4]

def click_red_then_blue(means3D, rotation, scales, opacity, colors, mask, time):
    tick = float((time*1800) % 270) #/270. # 9 sec
    if tick > 20 and tick < 60:

        colors[~mask] *= 0.
        colors[~mask, 0, 0] = 5.
    
    elif tick >= 60 and tick <= 100:
        colors[mask] *= 0.
        colors[mask, 0, 1] = 5.       
    else:
        pass

    return means3D, rotation, scales, opacity, colors, mask
 

def chromatic_abbr(means3D, rotation, scales, opacity, colors, mask, time):
    tick = float((time*1800) %360)/100.

    nn = 0.05
    meanR = means3D[~mask].clone() + torch.tensor([[nn,nn,nn]]).cuda()*tick
    meanB = means3D[~mask].clone() - torch.tensor([[nn,nn,nn]]).cuda()*tick
    colsA = colors[~mask].clone() 
    colsB = colors[~mask].clone() 
    
    colsA[:, :, 0] += 0.5*tick
    colsB[:, :, 2] += 0.5*tick
    
    
    means3D = torch.cat([means3D, meanR, meanB], dim=0)
    colors = torch.cat([colors, colsA, colsB], dim=0)
    
    rotation = torch.cat([rotation, rotation[~mask], rotation[~mask]], dim=0)
    opacity = torch.cat([opacity, opacity[~mask], opacity[~mask]], dim=0)
    scales = torch.cat([scales, scales[~mask], scales[~mask]], dim=0)
    mask = torch.cat([mask, mask[~mask], mask[~mask]], dim=0)

    return means3D, rotation, scales, opacity, colors, mask

def bloom(
    means3D, rotation, scales, opacity, colors_, mask, time,
    threshold=0.7,            # luminance threshold in [0,1]
    scale_factor=1.5,         # max scale multiplier for bright points
):
    tick = float((time*1800)) #/270. # 9 sec
    # if tick > 20 and tick < 60:
    
    colors = colors_.clone()
    device = colors.device
    dtype  = colors.dtype

    scales[~mask] = 0.*scales[~mask] + 0.05
    # --- Brightness from DC (l=0) SH term ---
    dc = colors[:, 0, :]  # (N, 3)
    # Perceived luminance weights, kept on same device/dtype
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=dtype)
    brightness = (dc * w).sum(dim=-1)  # (N,)

    # --- Build a smooth intensity in [0,1] only for unmasked points ---
    # Intensity grows linearly beyond threshold, zero otherwise.
    # Also zero-out masked points (your convention is to act on ~mask).
    raw = (brightness - threshold) / max(1e-6, 1.0 - threshold)
    intensity = torch.clamp(raw, min=0.0, max=1.0)
    intensity = intensity * (~mask).to(dtype)  # zero if masked

    # --- Scale splat size ---
    # Multiplier is 1 for non-bright points, up to scale_factor for very bright ones.
    scale_mult_1d = 1.0 + (scale_factor - 1.0) * intensity  # (N,)
    scales[~mask] = scales[~mask] * scale_mult_1d[~mask].unsqueeze(-1)

    colors[~mask] *= 10.
    colors[~mask,:, 0] = 10.
    colors[~mask,:, 1] = 5.
    colors[~mask,:, 2] = 2.5
    colors[~mask,1:] *= 0.
    
    # colors = (tick)*colors  + colors_*(1.-tick)
    
    return means3D, rotation, scales, opacity, colors, mask

def float_up(means3D, rotation, scales, opacity, colors, mask, time, global_rand):
    
    # tick is time for effect 0 to 1
    tick = float((time*1800) % 700)/120.
    current_radius = tick
     
    distance = torch.norm((means3D.mean(0).unsqueeze(0)- means3D).abs(), dim=1)
        
    dist_norm  = (distance - distance.min())/(distance.max()- distance.min())
    mask_r = (dist_norm < current_radius) & (~mask) # grow with tick
    dist_rad = current_radius - dist_norm
    # Function for determining position at current tick
    k = (10.*global_rand[:,0])
    
    # print(k.shape)
    delta_x = - torch.log(1-(dist_rad)**k)
    means3D[mask_r, 0] += delta_x[mask_r]

    # scales[mask_r] = (1.-tick)*scales[mask_r]
    
    colors[mask_r] *= 10.
    colors[mask_r,:, 0] = 10.
    colors[mask_r,:, 1] = 5.
    colors[mask_r,:, 2] = 2.5
    colors[mask_r,1:] *= 0.
    
    a = 4.5
    m = 10
    
    tick = dist_rad[mask_r].unsqueeze(-1) #(dist_rad + ((tick+0.1)*global_rand[mask_r])/10.)
    scales[mask_r] = (torch.exp(-a*tick)*torch.cos(m*tick)*scales[mask_r]).abs()
    
    return means3D, rotation, scales, opacity, colors, mask

def ripple_on(means3D, rotation, scales, opacity, colors, mask, time):
    
    # tick is time for effect 0 to 1
    tick = float((time*1800) % 90)/90.
    current_radius = tick
    
    distance = torch.norm((means3D.mean(0).unsqueeze(0)- means3D).abs(), dim=1)
        
    dist_norm  = (distance - distance.min())/(distance.max()- distance.min())
    mask_r = (dist_norm < current_radius) & (~mask) # grow with tick

    scales[mask_r] = (1.-tick)*scales[mask_r]
    
    colors[mask_r] *= 10.
    colors[mask_r,:, 0] = 10.
    colors[mask_r,:, 1] = 5.
    colors[mask_r,:, 2] = 2.5
    colors[mask_r,1:] *= 0.
    return means3D, rotation, scales, opacity, colors, mask


def nonlinfade(means3D, rotation, scales, opacity, colors, mask, time, global_rand):
    
    # tick is time for effect 0 to 1
    tick = float((time*1800) % 570)/130.
            
    mask_r = (~mask) # grow with tick

    a = 4.5
    m = 39
    
    tick = (tick + (tick*global_rand[mask_r])/10.)
    scales[mask_r] = torch.exp(-a*tick)*torch.cos(m*tick)*scales[mask_r]
    
    # colors[mask_r] *= 10.
    # colors[mask_r,:, 0] = 10.
    # colors[mask_r,:, 1] = 5.
    # colors[mask_r,:, 2] = 2.5
    # colors[mask_r,1:] *= 0.
    
    return means3D, rotation, scales, opacity, colors, mask

def render_cool_video(viewpoint_camera, pc, audio, global_rand):
    """
    Render the scene.
    """
    # local_randn = torch.rand_like(pc.get_scaling_with_3D_filter).cuda()

    extras = None

    means3D_ = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D_.device).repeat(means3D_.shape[0], 1)
    # colors = pc.get_color
    colors = pc.get_features

    opacity = pc.get_opacity.clone().detach()

    scales = pc.get_scaling_with_3D_filter
    rotations = pc._rotation

    means3D, rotations, opacity, colors, extras = pc._deformation(
        point=means3D_, 
        rotations=rotations,
        scales=scales,
        times_sel=torch.tensor((((viewpoint_camera.time*1800)% 600 ))/600.).to(means3D_.device).repeat(means3D_.shape[0], 1), 
        h_emb=opacity,
        shs=colors,
        view_dir=None,
        target_mask=pc.target_mask
    )
    
    opacity = pc.get_fine_opacity_with_3D_filter(opacity)
    rotation = pc.rotation_activation(rotations)
    
    # Clean the view
    distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
    mask = distances < .9
    mask = torch.logical_and(~pc.target_mask, mask)
    mask = ~mask
    means3D = means3D[mask]
    rotation = rotation[mask]
    scales = scales[mask]
    opacity = opacity[mask]
    colors = colors[mask]
    global_rand = global_rand[mask]
    pc_mask = pc.target_mask[mask]
    
    ##### Dealing with the view rotation #####
    viewmat = viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda()
    cent = means3D[pc_mask].mean(0).unsqueeze(0)
    K = viewpoint_camera.intrinsics.unsqueeze(0).cuda()
    W,H = viewpoint_camera.image_width, viewpoint_camera.image_height
    viewmat = rotate_view(viewmat, viewpoint_camera, cent) #offset=290)
    
    # means3D, rotation, scales, opacity, colors, pc_mask = click_red_then_blue(means3D, rotation, scales, opacity, colors, pc_mask, viewpoint_camera.time)

    # means3D, rotation, scales, opacity, colors, pc_mask = chromatic_abbr(means3D, rotation, scales, opacity, colors, pc_mask, viewpoint_camera.time)
    # means3D, rotation, scales, opacity, colors, pc_mask = bloom(means3D, rotation, scales, opacity, colors, pc_mask, viewpoint_camera.time)
    # means3D, rotation, scales, opacity, colors, pc_mask = ripple_on(means3D, rotation, scales, opacity, colors, pc_mask, viewpoint_camera.time)
    # means3D, rotation, scales, opacity, colors, pc_mask = float_up(means3D, rotation, scales, opacity, colors, pc_mask, viewpoint_camera.time, global_rand)
    # means3D, rotation, scales, opacity, colors, pc_mask = nonlinfade(means3D, rotation, scales, opacity, colors, pc_mask, viewpoint_camera.time, global_rand)
    means3D, rotation, scales, opacity, colors, pc_mask = float_up(means3D, rotation, scales, opacity, colors, pc_mask, viewpoint_camera.time, global_rand)


    
    rendered_image, alpha, _ = rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        viewmat,
        K,
        W,
        H,
        rasterize_mode='antialiased',
        eps2d=0.1,
        sh_degree=pc.active_sh_degree
    )

    rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    extras = alpha.squeeze(0).permute(2,0,1)

        
    return {
        "render": rendered_image,
        "extras":extras # A dict containing mor point info
        # 'norms':rendered_norm, 'alpha':rendered_alpha
        }

def fix_camera_global_x(viewmat, x_fixed):
    """
    viewmat: [B,4,4] world->camera
    x_fixed: float or tensor broadcastable to [B]
    returns: adjusted viewmat with camera center's global X locked
    """
    assert viewmat.shape[-2:] == (4, 4), "Expected [B,4,4]"

    B = viewmat.shape[0]
    device = viewmat.device
    dtype = viewmat.dtype

    R = viewmat[:, :3, :3]                 # [B,3,3]
    t = viewmat[:, :3, 3:4]                # [B,3,1]

    # camera center in world coords: C = -R^T t
    C = -(R.transpose(1, 2) @ t)           # [B,3,1]

    # set global X
    x_fixed = torch.as_tensor(x_fixed, device=device, dtype=dtype).view(-1, *([1]*(C.ndim-1)))
    if x_fixed.numel() == 1:               # scalar -> broadcast
        x_fixed = x_fixed.expand(B, 1, 1)
    C[:, 0:1, :] = x_fixed                 # lock X

    # recompute translation: t_new = -R * C_new
    t_new = -(R @ C)                       # [B,3,1]

    view_new = viewmat.clone()
    view_new[:, :3, 3:4] = t_new
    return view_new

def project_to_yz_circle(points: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    # Extract y and z
    y = points[:, 1]
    z = points[:, 2]

    # Compute angle with respect to origin in YZ plane
    theta = torch.atan2(z, y)

    # Project to circle in YZ plane
    y_new = radius * torch.cos(theta)
    z_new = radius * torch.sin(theta)

    # Construct new points with x=0
    x_new = torch.zeros_like(y_new)
    projected_points = torch.stack((x_new, y_new, z_new), dim=1)
    
    return projected_points, theta
