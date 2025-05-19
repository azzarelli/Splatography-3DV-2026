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
from scene.gaussian_model import GaussianModel
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
        

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           stage="fine", view_args=None, G=None):
    """
    Render the scene.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    extras = None

    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
    colors = pc.get_color

    opacity = pc.get_opacity.clone().detach()

    scales = pc.get_scaling
    rotations = pc._rotation

    if view_args is not None:
        if view_args['viewer_status']:
            if view_args['set_w_flag']:
                opacity[:, 1] =  opacity[:, 1]*0. + view_args['set_w']
                

    means3D, rotations, opacity,colors, extras = pc._deformation(
        point=means3D, 
        rotations=rotations,
        scales=scales,
        times_sel=time, 
        h_emb=opacity,
        shs=colors,
        view_dir=viewpoint_camera.direction_normal(),
        target_mask=pc.target_mask
    )

    rotation = pc.rotation_activation(rotations)

    if view_args is not None:
        if view_args['viewer_status']:

            show_mask = view_args['show_mask'] 
            
            mask = (pc.get_wopac.abs() > view_args['w_thresh'])
            mask = torch.logical_and(mask, (pc.get_hopac > view_args['h_thresh']).squeeze(-1))

            mask = mask.squeeze(-1)
            if show_mask == 1:
                mask = torch.logical_and(pc.target_mask, mask)
            elif show_mask == -1:
                mask = torch.logical_and(~pc.target_mask, mask)
            
   
            if mask is not None:
                means3D = means3D[mask]
                means2D = means2D[mask]
                colors = colors[mask]
                opacity = opacity[mask]
                scales = scales[mask]
                rotation = rotation[mask]
                
            if view_args['full_opac']:
                opacity = torch.ones_like(opacity).cuda()
            
    rendered_image, radii, rendered_depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotation,
        cov3D_precomp=None)
    
    # rendered_image = linear_rgb_to_srgb(rendered_image)
    norm = rotated_softmin_axis_direction(rotation, scales)

    norms, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=norm,
        opacities=opacity,
        scales=scales,
        rotations=rotation,
        cov3D_precomp=None)
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        'norms':norms, 'alpha':rendered_depth,
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

from utils.loss_utils import l1_loss, l1_loss_masked,local_triplet_ranking_loss
def render_batch(
    viewpoint_cams, pc: GaussianModel, pipe, bg_color: torch.Tensor,scaling_modifier=1.0,
    stage="fine", iteration=0):
    """
    Render the scene.
    """
    extras = {}
    means3D = pc.get_xyz    
    scales = pc.get_scaling
    rotations = pc._rotation
    colors = pc.get_color

    
    
    L1 = 0.
    norm_loss = 0.
    depth_loss = 0.
    covloss = 0.
    radii_list = []
    visibility_filter_list = []
    viewspace_point_tensor_list = []
    norms = None

    distances = []

    time = torch.tensor(viewpoint_cams[0].time).to(means3D.device).repeat(means3D.shape[0], 1).detach()
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        time = time*0. +viewpoint_camera.time
        
        if "coarse" in stage:
            means3D_final, rotations_final, opacity_final, colors_final = means3D, rotations, pc.get_hopac, colors
        elif "fine" in stage:
            means3D_final, rotations_final, opacity_final, colors_final, norms = pc._deformation(
                point=means3D, 
                rotations=rotations,
                scales = scales,
                times_sel=time, 
                h_emb=pc.get_opacity,
                shs=colors,
                view_dir=viewpoint_camera.direction_normal(),
                target_mask=pc.target_mask,
            )
        
        
        
        # Do the scaling and rotation activation after deformation
        rotations_final = pc.rotation_activation(rotations_final)
        
        # As we take the NN from some random time step, lets re-calc it frequently
        if (iteration % 10 == 0 and idx == 0) or pc.target_neighbours is None:
            pc.update_neighbours(means3D_final[pc.target_mask])
        
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means2D = screenspace_points
        
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)


        rendered_image, radii, Dt = rasterizer(
            means3D=means3D_final,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_final,
            opacities=opacity_final,
            scales=scales,
            rotations=rotations_final,
            cov3D_precomp=None)
        
        # Set raddii for non-targets to 0
        radii = radii * 0
        
        target_rgb, target_radii, target_depth = rasterizer(
            means3D=means3D_final[pc.target_mask],
            means2D=means2D[pc.target_mask],
            shs=None,
            colors_precomp=colors_final[pc.target_mask],
            opacities=opacity_final[pc.target_mask],
            scales=scales[pc.target_mask],
            rotations=rotations_final[pc.target_mask],
            cov3D_precomp=None
            )

        radii[pc.target_mask] = target_radii
        
        gt_img = viewpoint_camera.original_image.cuda()
        
        L1 += l1_loss(rendered_image, gt_img)
        # weights = viewpoint_camera.weights.cuda().unsqueeze(0)

        # L1 += l1_loss(rendered_image, gt_img, weights)

        # covloss += pc.compute_covariance_loss(
        #     means3D_final[pc.target_mask],
        #     rotations_final[pc.target_mask],
        #     scales[pc.target_mask]
        # )
        
        if stage == 'coarse':
            mask = viewpoint_camera.mask.cuda()
            L1 += l1_loss(target_rgb, gt_img*mask)
            
        elif stage == 'fine' :
            # norms
            norm_loss += pc.compute_normals_rigidity(norms=norms)
            # distances.append(pc.compute_displacement_rigidity(means3D_final[pc.target_mask]))

        # # Depth pred
        # with torch.no_grad():
        #     De = viewpoint_camera.depth.cuda().squeeze(0)
        #     Demask = De > 0
        #     De[Demask] = (De[Demask] - De[Demask].min())/ (De[Demask].max() - De[Demask].min())
        
        # Dt = Dt.squeeze(0)
        # Dtmask = Dt > 0
        # Dt[Dtmask] = (Dt[Dtmask] - Dt[Dtmask].min())/ (Dt[Dtmask].max() - Dt[Dtmask].min())
        
        # dt_vals = Dt[Dtmask].flatten()
        # de_vals = De[Demask].flatten()
        # min_len = min(len(dt_vals), len(de_vals))
        # dt_vals = dt_vals[:min_len]
        # de_vals = de_vals[:min_len]

        # # Apply differentiable remapping
        # Dt[Dtmask] = differentiable_cdf_match(dt_vals, de_vals)
        # multiplier = (target_depth > 0).float().detach()+0.2
        # depth_loss += ((multiplier*(Dt - De)).abs()).mean()
       

        radii_list.append(radii.unsqueeze(0))
        visibility_filter_list.append((radii > 0).unsqueeze(0))
        viewspace_point_tensor_list.append(screenspace_points)
    
    # if stage == 'fine':
    #     distance_stack = torch.cat(distances, dim=-1)
    #     mean = distance_stack.mean(dim=-1).unsqueeze(-1)
    #     covloss += (distance_stack - mean).abs().mean()
    
    return radii_list,visibility_filter_list, viewspace_point_tensor_list, L1, (depth_loss, norm_loss, covloss)


def masked_mean(patches):
    # patches shape: (N, patch_size * patch_size)
    mask = (patches != 0).float()
    summed = (patches * mask).sum(dim=1, keepdim=True)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
    return summed / count
# def normalize(input, mean=None, std=None):
#     input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
#     input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
#     return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))
def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches
def minmax_normalize(patches, eps=1e-8):
    min_vals = patches.min(dim=1, keepdim=True)[0]
    max_vals = patches.max(dim=1, keepdim=True)[0]
    denom = (max_vals - min_vals).clamp(min=eps)
    norm_patches = (patches - min_vals) / denom
    return norm_patches
def unpatchify(patches, image_size, patch_size):
    # patches: (num_patches, patch_area)
    B = 1
    C = 1
    H, W = image_size
    num_patches = patches.size(0)
    patches = patches.view(B, -1, patch_size * patch_size).permute(0, 2, 1)
    output = F.fold(patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    # Correct for patch overlap (we assume no overlap here)
    return (output ).squeeze()

def render_batch_coarse_depth(
    viewpoint_cams, pc: GaussianModel, pipe, bg_color: torch.Tensor,scaling_modifier=1.0,
    stage="fine", iteration=0):
    """
    Render the scene.
    """
    extras = {}
    means3D = pc.get_xyz    
    scales = pc.get_scaling.detach()
    rotations = pc._rotation
    colors = pc.get_features.detach()

    opacity = pc.get_hopac.detach()

    
    # Do the scaling and rotation activation after deformation
    rotation = pc.rotation_activation(rotations).detach()
    
    L1 = 0.

    for viewpoint_camera in viewpoint_cams:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means2D = screenspace_points
        
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        target_rgb, _, target_depth = rasterizer(
            means3D=means3D[pc.target_mask],
            means2D=means2D[pc.target_mask],
            shs=colors[pc.target_mask],
            colors_precomp=None,
            opacities=torch.ones_like(opacity[pc.target_mask], dtype=torch.float, device=opacity.device),
            scales=scales[pc.target_mask],
            rotations=rotation[pc.target_mask],
            cov3D_precomp=None
            )

        patch_size = 16
        H,W = target_depth.shape[1], target_depth.shape[2]
        
        with torch.no_grad():
            target_mask = target_depth > 0.
            depth = target_depth.unsqueeze(0)
            mono_depth = viewpoint_camera.mask.cuda() * target_mask
            mono_depth = minmax_normalize(patchify(mono_depth.unsqueeze(0), patch_size))
            m = unpatchify(mono_depth,(H,W), patch_size)

        target_depth = minmax_normalize(patchify(target_depth.unsqueeze(0), patch_size))
        t = unpatchify(target_depth,(H,W), patch_size)

        # plot_depth_comparison(t, m)
        # exit()
        L1 += (t-m).abs().mean()
        # exit()
        
    return L1


def plot_depth_comparison(original, smoothed):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original.detach().squeeze().cpu(), cmap='plasma')
    axs[0].set_title('Original Depth')
    axs[0].axis('off')

    axs[1].imshow(smoothed.detach().squeeze().cpu(), cmap='plasma')
    axs[1].set_title('Smoothed Depth')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def render_depth(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    """
    Render the depth at t = 0
    """
    means3D = pc.get_xyz
    scales = pc.get_scaling
    rotations = pc._rotation
    colors = pc.get_color
    opacity = pc.get_opacity
    
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)


    means3D, rotations, opacity, colors, extras = pc._deformation(
        point=means3D, 
        rotations=rotations,
        scales=scales,
        times_sel=time, 
        h_emb=opacity,
        shs=colors,
        view_dir=viewpoint_camera.direction_normal(),
        target_mask=pc.target_mask
    )

    rotation = pc.rotation_activation(rotations)

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means2D = screenspace_points
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    return rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotation,
        cov3D_precomp=None)
