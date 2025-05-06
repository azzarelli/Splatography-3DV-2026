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

from scene.gaussian_model import GaussianModel


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
    colors = pc.get_features

    scales = pc.get_scaling
    rotations = pc._rotation

    means3D, rotations_final, opacity,colors, extras = pc._deformation(
        point=means3D, 
        rotations=rotations,
        scales=scales,
        times_sel=time, 
        h_emb=pc.get_opacity,
        shs=colors,
    )

    rotation = pc.rotation_activation(rotations_final)

    if view_args is not None:
        show_mask = view_args['show_mask'] 
        mask = None

        if mask is not None:
            means3D = means3D[mask]
            means2D = means2D[mask]
            colors = colors[mask]
            opacity = opacity[mask]
            scales = scales[mask]
            rotation = rotation[mask]
            
        if view_args['full_opac']:
            opacity = torch.ones_like(opacity).cuda()
    
    # colors = pc._deformation.deformation_net.get_view_color(colors, viewpoint_camera.R)

    rendered_image, radii, rendered_depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=colors,
        colors_precomp=None,
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
        'norms':rendered_image, 'alpha':rendered_depth,
        "extras":extras # A dict containing mor point info
        # 'norms':rendered_norm, 'alpha':rendered_alpha
        }

from utils.loss_utils import l1_loss, l1_loss_masked
def render_batch(viewpoint_cams, pc: GaussianModel, pipe, bg_color: torch.Tensor, means3D=None, scales=None, rotation=None, opacity=None, shs=None, scaling_modifier=1.0,
           stage="fine"):
    """
    Render the scene.
    """
    extras = {}
    
    if stage == 'coarse':
        means3D = pc.get_xyz    
        scales = pc.get_scaling
        rotations = pc._rotation
        opacity = pc.get_hopac
        colors = pc.get_features
    else:
        # freeze backprop to G representation
        # with torch.no_grad():
        means3D = pc.get_xyz    
        scales = pc.get_scaling
        rotations = pc._rotation
        colors = pc.get_features
        opacity = pc.get_opacity

    
    
    L1 = 0.
    radii_list = []
    visibility_filter_list = []
    viewspace_point_tensor_list = []

    for viewpoint_camera in viewpoint_cams:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0.
        
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1).detach()

        if "coarse" in stage:
            means3D_final, rotations_final, opacity_final, colors_final = means3D, rotations, opacity[:, 0].unsqueeze(-1), colors
        elif "fine" in stage:
            means3D_final, rotations_final, opacity_final, colors_final, _ = pc._deformation(
                point=means3D, 
                rotations=rotations,
                scales = scales,
                times_sel=time, 
                h_emb=opacity,
                shs=colors,
            )

        rotation_final = pc.rotation_activation(rotations_final)
        
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

    
        rendered_image, radii, rendered_depth = rasterizer(
            means3D=means3D_final,
            means2D=means2D,
            shs=colors_final,
            colors_precomp=None,
            opacities=opacity_final,
            scales=scales,
            rotations=rotation_final,
            cov3D_precomp=None)

        gt_img = viewpoint_camera.original_image.cuda()

        L1 += l1_loss(rendered_image, gt_img)

        
        radii_list.append(radii.unsqueeze(0))
        visibility_filter_list.append((radii > 0).unsqueeze(0))
        viewspace_point_tensor_list.append(screenspace_points)
    
    return radii_list,visibility_filter_list, viewspace_point_tensor_list, L1, extras
