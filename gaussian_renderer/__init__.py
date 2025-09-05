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

    pc_stage = 'fine'
    pc_return_type = 'full'  
    
    if stage == 'test-foreground':
        pc_return_type = 'foreground'
    
    elif view_args is not None:
        if view_args['finecoarse_flag']:
            pc_stage = 'fine'
        else:
            pc_stage = 'coarse'

        if view_args['show_mask'] == 1:
            pc_return_type = 'foreground'
        elif view_args['show_mask'] == -1:
            pc_return_type = 'background'


    means3D, rotation,opacity, colors, scales = process_gaussians(pc, viewpoint_camera.time, pc_stage, pc_return_type)
            
    rendered_image, rendered_depth, norms = None, None, None
    if stage == 'test-foreground':
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

from utils.loss_utils import l1_loss


def process_gaussians(pc, time, stage, return_type='full'): # return_type: [full, foreground, background]
    """Process the Gaussians depending on our stage
    """
    means = pc.get_xyz
    scales = pc.get_scaling_with_3D_filter

    pts, rotations, opacity, shs = pc._deformation(
        point=means, 
        time=time,
        target_mask=pc.target_mask,
        stage=stage,
        return_type=return_type
    )
    
    opacity = pc.get_fine_opacity_with_3D_filter(opacity)
    rotations = pc.rotation_activation(rotations)
    
    if return_type == 'full':
        scales = torch.cat([scales[pc.target_mask], scales[~pc.target_mask]], dim=0)
    elif return_type == 'foreground':
        scales = scales[pc.target_mask]
    elif return_type == 'background':
        scales = scales[~pc.target_mask]
    else:
        raise f'Issue with return_type in process_gaussians in the renderer'
    
    
    shs = shs.view(-1, 16, 3)
    return pts, rotations, opacity, shs, scales
        
def render_coarse_background(viewpoint_cams, pc, kernel_size=0.1):
    """
    Background Rendering and Loss
    """
    
    means3D, rotations,opacity, colors, scales = process_gaussians(pc,-1, 'coarse', 'background')

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

def render_coarse_foreground(viewpoint_cams, pc):
    """
    Foreground Rendering and Loss
    """

    means3D, rotations, opacity, colors, scales = process_gaussians(pc, -1, 'coarse', 'foreground')

    L1 = 0.
    time = torch.tensor(viewpoint_cams[0].time).to(means3D.device).repeat(means3D.shape[0], 1).detach()
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        time = time*0. + viewpoint_camera.time

        background = torch.rand(1, 3).cuda() 
        rgb, _, _ = rasterization(
            means3D, rotations, scales, 
            opacity.squeeze(-1), colors,
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
    L1 = 0.

    for idx, viewpoint_camera in enumerate(viewpoint_cams):        
        means3D_final, rotations_final,opacity_final, colors_final, scales_final = process_gaussians(pc, viewpoint_camera.time, 'fine', 'full')

        # For ViVo mask out near by Gaussians that may be distractors
        if datasettype == 'condense':
            distances = torch.norm(means3D_final - viewpoint_camera.camera_center.cuda(), dim=1)
            mask = distances > 0.3
            means3D_final = means3D_final[mask]
            rotations_final = rotations_final[mask]
            scales_final = scales_final[mask]
            opacity_final = opacity_final[mask]
            colors_final = colors_final[mask]
        
        # Set up rasterization configuration
        rgb, _, _ = rasterization(
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

    
    L1 = 0.

    time = torch.zeros_like(means3D[:, 0], device=means3D.device).unsqueeze(-1)
    means3D_collection = []
    for i in range(10):
        time = time*0. + float(i)*0.1

        means3D_final, rotations_final,opacity_final, colors_final, scales_final = process_gaussians(pc, time, 'fine', 'full')

        
        means3D_collection.append(means3D_final[pc.target_mask].unsqueeze(0))
    
    means3D_collection = torch.cat(means3D_collection, dim=0) # K, N, 3, where K=10 for each time step
    displacement = ((means3D_collection - means3D_collection.mean(dim=0))**2).sum(dim=2).sqrt()  # K, N, 3
    motion_metric = displacement.mean(dim=0) # shape (N,)

    threshold = torch.quantile(motion_metric, 0.9)

    mask = (motion_metric >= threshold)
    
    final_mask = torch.zeros_like(pc.target_mask, device=means3D.device)
    final_mask[pc.target_mask] = mask
    return final_mask

