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


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, means3D=None, scales=None, rotation=None, opacity=None, shs=None, scaling_modifier=1.0,
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

    if means3D is None:
        means3D = pc.get_xyz
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
        shs = pc.get_features


        scales = pc._scaling
        rotations = pc._rotation

        h_opacity = pc.get_h_opacity

        if "coarse" in stage:
            means3D, rotations_final, opacity, shs = means3D, rotations, h_opacity, shs

            # means3D_final, scales_final, rotations_final, opacity, shs_final = means3D, scales, rotations, torch.ones_like(means3D[..., 0]), shs
        elif "fine" in stage:
            means3D, rotations_final, opacity, shs, extras = pc._deformation(
                point=means3D, 
                rotations=rotations,
                shs=shs,
                times_sel=time, 
                h_emb=h_opacity,
                target_mask=pc.target_mask
            )
        else:
            raise NotImplementedError
        # Do the scaling and rotation activation after deformation
        scales = pc.scaling_activation(scales + 0.)
        rotation = pc.rotation_activation(rotations_final)
    
    else:
        means3D, scales, rotation, opacity, shs = pc.get_G(viewpoint_camera.time, stage)

    
    if view_args is not None:
        show_mask = view_args['show_mask'] 
        mask = None
        if show_mask == 1:
            mask = pc.target_mask
        elif show_mask == -1:
            mask = ~pc.target_mask

        if mask is not None:
            means3D = means3D[mask]
            means2D = means2D[mask]
            shs = shs[mask]
            opacity = opacity[mask]
            scales = scales[mask]
            rotation = rotation[mask]
    
    rendered_image, radii, rendered_depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
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

from utils.loss_utils import l1_loss
def render_batch(viewpoint_cams, pc: GaussianModel, pipe, bg_color: torch.Tensor, means3D=None, scales=None, rotation=None, opacity=None, shs=None, scaling_modifier=1.0,
           stage="fine"):
    """
    Render the scene.
    """
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_cams[0].time).to(means3D.device).repeat(means3D.shape[0], 1)
    shs = pc.get_features

    scales = pc._scaling
    rotations = pc._rotation

    h_opacity = pc.get_h_opacity

    if "coarse" in stage:
        means3D, rotations_final, opacity, shs = means3D, rotations, h_opacity, shs

        # means3D_final, scales_final, rotations_final, opacity, shs_final = means3D, scales, rotations, torch.ones_like(means3D[..., 0]), shs
    elif "fine" in stage:
        means3D, rotations_final, opacity, shs, extras = pc._deformation(
            point=means3D, 
            rotations=rotations,
            shs=shs,
            times_sel=time, 
            h_emb=h_opacity,
            target_mask=pc.target_mask
        )
    else:
        raise NotImplementedError
    # Do the scaling and rotation activation after deformation
    scales = pc.scaling_activation(scales + 0.)
    rotation = pc.rotation_activation(rotations_final)
    
    # else:
    #     means3D, scales, rotation, opacity, shs = pc.get_G(viewpoint_camera.time, stage)
    L1 = 0.
    radii_list = []
    visibility_filter_list = []
    viewspace_point_tensor_list = []

    for viewpoint_camera in viewpoint_cams:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
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
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotation,
            cov3D_precomp=None)
        
        L1 += l1_loss(rendered_image, viewpoint_camera.original_image.cuda())
        
        radii_list.append(radii.unsqueeze(0))
        visibility_filter_list.append((radii > 0).unsqueeze(0))
        viewspace_point_tensor_list.append(screenspace_points)
    
    return radii_list,visibility_filter_list, viewspace_point_tensor_list, L1




def render_hard(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           stage="fine"):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    means3D = pc.get_xyz
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
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    cov3D_precomp = None
    
    with torch.no_grad():
        scales = pc.get_scaling.detach()
        rotations = pc._rotation.detach()
        
        opacity = torch.ones(pc.get_xyz.shape[0], 1, device=pc.get_xyz.device) * 0.95
        
    if "coarse" in stage:
        means3D_final = means3D
    
    elif "fine" in stage:
        means3D_final, rotations, _, _, _ = pc._deformation(
            point=means3D, 
            rotations=rotations,
            times_sel=time, 
            target_mask=pc.target_mask
        )
    else:
        raise NotImplementedError


    # Do the scaling and rotation activation after deformation
    with torch.no_grad():
        rotations = pc.rotation_activation(rotations)
        

    rendered_image, radii, rendered_depth = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=None,
        colors_precomp=torch.ones_like(pc.get_xyz, device=pc.get_xyz.device),
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        }
    
def render_soft(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           stage="fine"):
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
    means3D = pc.get_xyz.detach()

    with torch.no_grad():
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
        
        scales = pc.get_scaling.detach()
        rotations = pc._rotation.detach()

    opacity = pc.get_h_opacity
    
    if "fine" in stage:
        means3D, rotations, opacity, _, _ = pc._deformation(
            point=means3D, 
            rotations=rotations,
            h_emb=opacity,
            times_sel=time, 
            target_mask=pc.target_mask
        )



    # Do the scaling and rotation activation after deformation
    rotations = pc.rotation_activation(rotations)
        

    rendered_image, radii, rendered_depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=torch.ones_like(pc.get_xyz, device=pc.get_xyz.device),
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        }