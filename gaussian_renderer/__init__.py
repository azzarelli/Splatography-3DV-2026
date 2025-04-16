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
           stage="fine", view_args=None, ):
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
    shs = pc.get_features

    cov3D_precomp = None
    colors_precomp=None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier) 
    else:
        scales_final = pc.get_scaling
        rotations = pc._rotation


    # rasterizer.markVisibleMasked(viewpoint_camera.original_image.cuda(),means3D)
    # exit()
    extras = None
    if "coarse" in stage:
        means3D_final, rotations_final, opacity, shs_final = means3D, rotations, torch.ones_like(means3D[:,0], device=means3D.device), shs

        # means3D_final, scales_final, rotations_final, opacity, shs_final = means3D, scales, rotations, torch.ones_like(means3D[..., 0]), shs
    elif "fine" in stage:
        means3D_final, _, rotations_final, opacity, shs_final, extras = pc._deformation(means3D, scales_final,
                                                                                                 rotations,
                                                                                                 shs,
                                                                                                 time)
    else:
        raise NotImplementedError

    # Do the scaling and rotation activation after deformation
    # scales_final = pc.scaling_activation(scales)
    rotations_final = pc.rotation_activation(rotations_final)

    if view_args is not None:
        # show_radius = view_args['show_radius']
        # dx_thresh = view_args['dx_prob']
        # w_thresh = view_args['w']
        # h_thresh = view_args['h']
        
        
        
        # distances = torch.norm(means3D_final, dim=1)
        # Create a mask for the bounding box
        # print(view_args['mask'].max())
        mask = view_args['mask'] 
        # print(mask.max())
        # exit()
        # if stage == 'fine':
        #     w,h,_ = pc.get_cached_opacity
        #     mask = mask * (h.squeeze(-1) > h_thresh) * (w.squeeze(-1).abs()  < w_thresh)
        #     mask = mask * (pc.dynamic_point_prob.squeeze(-1) > dx_thresh)
        
        means3D_final = means3D_final[mask]
        means2D = means2D[mask]
        shs_final = shs_final[mask]
        opacity = opacity[mask]
        scales_final = scales_final[mask]
        rotations_final = rotations_final[mask]
        # print(means3D_final.shape)
        # pass
        
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
    #     means3D=means3D_final,
    #     means2D=means2D,
    #     shs=shs_final,
    #     colors_precomp=colors_precomp,
    #     opacities=opacity,
    #     scales=scales_final,
    #     rotations=rotations_final,
    #     cov3Ds_precomp=cov3D_precomp)

    rendered_image, radii, rendered_depth = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=shs_final,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp)

    # exit()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        'norms':rendered_image, 'alpha':rendered_depth,
        "extras":extras
        # 'norms':rendered_norm, 'alpha':rendered_alpha
        }
