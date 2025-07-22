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

from gsplat.rendering import rasterization
from utils.loss_utils import l1_loss

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

@torch.no_grad
def render(viewpoint_camera, pc, datasettype, stage="fine", view_args=None):
    

    extras = None
    

    if view_args is not None:
        if view_args['finecoarse_flag']:
            means3D, scales, rotation, colors, opacity = pc.get_temporal_parameters(
                viewpoint_camera, datasettype
            )

        else:
            means3D, scales, rotation, colors, opacity = pc.get_full_canonical_parameters()

    else:
        means3D, scales, rotation, colors, opacity = pc.get_temporal_parameters(
            viewpoint_camera, datasettype
        )
    

    show_mask = 0
    if view_args is not None and 'test' not in stage:
        if view_args['viewer_status']:
            pass
            # show_mask = view_args['show_mask'] 
            
            # mask = (pc._opacityw.abs() > view_args['w_thresh'])
            # mask = torch.logical_and(mask, (pc._opacityh > view_args['h_thresh']))

            # mask = mask.squeeze(-1)

            # if show_mask == 1:
            #     mask = torch.logical_and(pc.target_mask, mask)
            # elif show_mask == -1:
            #     mask = torch.logical_and(~pc.target_mask, mask)
            
            # if mask is not None:
            #     means3D = means3D[mask]
            #     colors = colors[mask]
            #     opacity = opacity[mask]
            #     scales = scales[mask]
            #     rotation = rotation[mask]
                
            # if view_args['full_opac']:
            #     opacity = torch.ones_like(opacity).cuda()
            #     colors = (means3D - means3D_).abs()
    else:
        view_args= {'vis_mode':'render'}
        
        
    # print(.shape, means3D.shape)
    rendered_image, rendered_depth, norms = None, None, None
    if stage == 'test-foreground':
        distances = torch.norm(means3D - viewpoint_camera.camera_center.cuda(), dim=1)
        mask = distances > 0.3
        mask = torch.logical_and(pc.target_mask, mask)

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
            sh_degree=pc.fg.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    elif stage == 'test-full':
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
            sh_degree=pc.fg.active_sh_degree
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
            sh_degree=pc.fg.active_sh_degree
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
            sh_degree=pc.fg.active_sh_degree
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
            sh_degree=pc.fg.active_sh_degree
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
            sh_degree=pc.fg.active_sh_degree
        )
        rendered_image = (rendered_image - rendered_image.min())/ (rendered_image.max() - rendered_image.min())
        rendered_image = rendered_image.squeeze(0).permute(2,0,1).repeat(3,1,1)
    elif view_args['vis_mode'] == 'norms':
        norms = rotated_softmin_axis_direction(rotation, scales)

        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),norms,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.fg.active_sh_degree
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
            # sh_degree=pc.fg.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    elif view_args['vis_mode'] == 'dxyz_1':
        residual = torch.norm(means3D, dim=-1).unsqueeze(-1).repeat(1,3)
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),residual,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.fg.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
    elif view_args['vis_mode'] == 'dxyz_3' or view_args['vis_mode'] == 'extra':
        residual = (means3D).abs()
        rendered_image, _, _ = rasterization(
            means3D, rotation, scales, opacity.squeeze(-1),residual,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            # sh_degree=pc.fg.active_sh_degree
        )
        rendered_image = rendered_image.squeeze(0).permute(2,0,1)
  
    return {
        "render": rendered_image,
        "extras":extras # A dict containing mor point info
        # 'norms':rendered_norm, 'alpha':rendered_alpha
        }

def render_coarse_background(viewpoint_cams, pc, kernel_size=0.1):
    L1 = 0.
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        means3D_final, scales_final, rotations_final, colors_final, opacity_final = pc.bg.get_canonical_parameters(
            mask='distance', 
            cam_center=viewpoint_camera.camera_center.cuda(), 
            distance=0.3
        )

        rgb, _, _ = rasterization(
            means3D_final, rotations_final, scales_final, 
            opacity_final.squeeze(-1),colors_final,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.fg.active_sh_degree
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
    L1 = 0.
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        means3D_final, scales_final, rotations_final, colors_final, opacity_final = pc.fg.get_canonical_parameters()
        
        background = torch.rand(1, 3).cuda() 
        rgb, _, _ = rasterization(
            means3D_final, rotations_final, scales_final, 
            opacity_final.squeeze(-1), colors_final,
            viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0).cuda(), 
            viewpoint_camera.intrinsics.unsqueeze(0).cuda(),
            viewpoint_camera.image_width, 
            viewpoint_camera.image_height,
            
            rasterize_mode='antialiased',
            eps2d=0.1,
            sh_degree=pc.fg.active_sh_degree,
            backgrounds=background
        )
        
        rgb = rgb.squeeze(0).permute(2,0,1)
                
        gt = viewpoint_camera.original_image.cuda()
        mask = viewpoint_camera.mask.cuda() # > 0. # invert binary mask
        gt = gt*mask + (1.-mask)*background.permute(1,0).unsqueeze(-1)
        
        L1 += l1_loss(rgb, gt)
    return  L1



def render_batch(viewpoint_cams, pc, datasettype):
    
    L1 = 0.
    for idx, viewpoint_camera in enumerate(viewpoint_cams):
        means3D_final, scales_final, rotations_final, colors_final, opacity_final = pc.get_temporal_parameters(
            viewpoint_camera, datasettype
        )
        
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
            sh_degree=pc.fg.active_sh_degree
        )
        rgb = rgb.squeeze(0).permute(2,0,1)
        gt_img = viewpoint_camera.original_image.cuda()

        # For ViVo
        if datasettype == 'condense': # remove black edge from loss (edge from undistorting the images)
            L1 += l1_loss(rgb[:, 100:-100, 100:-100], gt_img[:, 100:-100, 100:-100])
        else:
            L1 += l1_loss(rgb, gt_img)
   
    return L1

def render_motion_point_mask(pc):
    means3D_collection = []
    for i in range(10):
        time = float(i)*0.1

        means3D_final = pc.get_position_changes(time)
        means3D_collection.append(means3D_final.unsqueeze(0))
    
    means3D_collection = torch.cat(means3D_collection, dim=0) # K, N, 3, where K=10 for each time step
    displacement = ((means3D_collection - means3D_collection.mean(dim=0))**2).sum(dim=2).sqrt()  # K, N, 3
    motion_metric = displacement.mean(dim=0) # shape (N,)

    threshold = torch.quantile(motion_metric, 0.9)

    mask = (motion_metric >= threshold)
    
    return mask
