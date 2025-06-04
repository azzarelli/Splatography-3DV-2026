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

# AbsGS for better split on larger points split from https://github.com/TY424/AbsGS/blob/main/scene/gaussian_model.py

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness,compute_plane_tv

from torch_cluster import knn_graph

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.color_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        
        self._opacity = torch.empty(0)
        self._colors = torch.empty(0)

        self._deformation = deform_network(args)
        
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.target_mask = None
        self.target_neighbours = None
        
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._features_dc,
            self._features_rest,
            self._colors,
            self._scaling,
            self._rotation,
            self._opacity,
            self.filter_3D,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs ,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.target_mask
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        deform_state,
        self._features_dc,
        self._features_rest,
        self._colors,
        self._scaling,
        self._rotation,
        self._opacity,
        self.filter_3D,
        self.max_radii2D,
        xyz_gradient_accum,
        xyz_gradient_accum_abs,
        denom,
        opt_dict,
        self.spatial_lr_scale, self.target_mask) = model_args

        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs  = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_color(self):
        return self.color_activation(self._colors)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    def get_fine_opacity_with_3D_filter(self, opacity):
        scales = self.get_scaling
        filter3D = self.filter_3D
        if opacity.shape[0] != scales.shape[0]:
            scales = scales[self.target_mask]
            filter3D = filter3D[self.target_mask]
            
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(filter3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz


    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_coarse_opacity_with_3D_filter(self):
        opacity = torch.sigmoid(self.get_opacity[:, 0]).unsqueeze(-1)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    @property
    def get_hopac(self):
        return torch.sigmoid(self.get_opacity[:, 0]).unsqueeze(-1)
    @property
    def get_wopac(self):
        return self.get_opacity[:, 1]
    @property
    def get_muopac(self):
        return torch.sigmoid(self.get_opacity[:, 2]).unsqueeze(-1)
    
    @property
    def get_dyn_coefs(self):
        return self._deformation.get_dyn_coefs(self.get_xyz[self.target_mask],self.get_scaling[self.target_mask])
    
    @property
    def dynamic_point_prob(self):
        return self._deformation.deformation_net.grid.get_dynamic_probabilities(self._xyz)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.1
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]
        
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int, cam_list=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        # Construct pcd, increase number of points in target region
        # increase scene points as well
        # Remove unseen points
        # Remove points betwen 1 and  target view masks

        if False: # This is for the dynerf dataset
            dyn_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()
            scene_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()
            
            for cam in cam_list:             
                dyn_mask += get_in_view_dyn_mask(cam, fused_point_cloud).long()
                scene_mask += get_in_view_screenspace(cam, fused_point_cloud).long()
            
            scene_mask = scene_mask > 0
            target_mask = dyn_mask > (len(cam_list)-1)
            dyn_mask = torch.logical_or(target_mask, dyn_mask ==0)
            viable  = torch.logical_and(dyn_mask, scene_mask)
        else:
            target_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()

            # For the condense dataset we use the bounding box
            # Just for the basssist test scene
            CORNER_2 = [[-1.38048, -0.1863],[-0.7779, 1.6705], [1.1469, 1.1790], [0.5832, -0.7245]]
            polygon = np.array(CORNER_2)  # shape (4, 2)
            from matplotlib.path import Path
            path = Path(polygon)
            points_xy = fused_point_cloud[:, 1:].cpu().numpy()  # (N, 2)
            # Create mask for points inside polygon
            viable = torch.from_numpy(path.contains_points(points_xy)).cuda()
            
            
        # Downsample background gaussians
        pcds = fused_point_cloud[~viable].cpu().numpy().astype(np.float64)
        cols = fused_color[~viable].cpu().numpy().astype(np.float64)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcds)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        # Voxel size controls the granularity
        voxel_size = 0.05  # Adjust based on your data scale
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        # Convert back to PyTorch tensor
        ds_pcd = torch.tensor(np.asarray(downsampled_pcd.points), dtype=fused_point_cloud.dtype).cuda()
        ds_cols = torch.tensor(np.asarray(downsampled_pcd.colors), dtype=fused_color.dtype).cuda()
        
        # for cam in cam_list:
        #     ds_pcd, ds_cols = populate_background(cam, ds_pcd, ds_cols)
        
        # Re-sample point cloud
        target = fused_point_cloud[viable]
        target_col = fused_color[viable]
        
        # for cam in cam_list:
        #     target, target_col = get_in_view_dyn_mask(cam, target, target_col)
            
            # points_xy = target[:, 1:].cpu().numpy()  # (N, 2)
            # # Create mask for points inside polygon
            # viable = torch.from_numpy(path.contains_points(points_xy)).cuda()
            
            # target = target[viable]
            # target_col = target_col[viable]
        
        # for cam in cam_list:
        #     target, target_col = refilter_pcd(cam, target, target_col)
            
        fused_point_cloud = torch.cat([ds_pcd, target], dim=0)
        fused_color = torch.cat([ds_cols, target_col], dim=0)
        target_mask = torch.zeros((fused_color.shape[0], 1)).cuda()
        target_mask[ds_cols.shape[0]:, :] = 1
        target_mask = (target_mask > 0.).squeeze(-1)
    
        
        # while target_mask.sum() < 40000:
        #     target_point_noise =  fused_point_cloud[target_mask] + torch.randn_like(fused_point_cloud[target_mask]).cuda() * 0.05
        #     fused_point_cloud = torch.cat([fused_point_cloud,target_point_noise], dim=0)
        #     fused_color = torch.cat([fused_color,fused_color[target_mask]], dim=0)
        #     target_mask = torch.cat([target_mask, target_mask[target_mask]])
        
        self.target_mask = target_mask
        
        # Prune background down to 100k
        xyz_min = fused_point_cloud[target_mask].min(0).values - .05
        xyz_max = fused_point_cloud[target_mask].max(0).values + .05
        self._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
        
        xyz_min = fused_point_cloud[~target_mask].min(0).values
        xyz_max = fused_point_cloud[~target_mask].max(0).values
        self._deformation.deformation_net.set_aabb(xyz_max, xyz_min, grid_type='background')
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # fused_color[~target_mask] = fused_color[~target_mask] + torch.clamp(torch.rand(fused_color[~target_mask].shape[0], 3).cuda()*0.1, 0., 1.)
        
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud[target_mask]), 0.00000000001)
        dist2_else = torch.clamp_min(distCUDA2(fused_point_cloud[~target_mask]), 0.00000000001)
        dist2 = torch.cat([dist2_else, dist2], dim=0)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Initialize opacities
        opacities = 1. * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        # Set h = 1 : As max_opac = sig(h) to set max opac = 1 we need h = logit(1)
        opacities[:, 0] = torch.logit(opacities[:, 0]*0.95)
        # Set w = 0.01 : As w_t = sig(w)*200, we need to set w = logit(w_t/200)
        opacities[:, 1] = (opacities[:, 1]*1.5)
        # Finally set mu to 0 as the start of the traniing
        opacities[:, 2] = torch.logit(opacities[:, 2]*0.5)
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._colors = nn.Parameter(fused_color.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        self._deformation = self._deformation.to("cuda")
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # self.active_sh_degree = 0
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},            
            # {'params': [self._colors], 'lr': training_args.feature_lr, "name": "color"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        # for i in range(self._colors.shape[1]):
        #     l.append('color_{}'.format(i))
        for i in range(self._opacity.shape[1]):
            l.append('opacity_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l
    
    # def compute_deformation(self,time):
    #     deform = self._deformation[:,:,:time].sum(dim=-1)
    #     xyz = self._xyz + deform
    #     return xyz

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        scale = self._scaling.detach().cpu().numpy()
        colors = self._colors.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals, colors, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opac_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity")]
        opac_names = sorted(opac_names, key = lambda x: int(x.split('_')[-1]))
        opacities = np.zeros((xyz.shape[0], len(opac_names)))
        for idx, attr_name in enumerate(opac_names):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        col_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("color")]
        col_names = sorted(col_names, key = lambda x: int(x.split('_')[-1]))
        cols = np.zeros((xyz.shape[0], len(col_names)))
        for idx, attr_name in enumerate(col_names):
            cols[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._colors = nn.Parameter(torch.tensor(cols, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                param = group["params"][0]
                
                # Ensure the optimizer state is initialized
                if param not in self.optimizer.state:
                    self.optimizer.state[param] = {}
                stored_state = self.optimizer.state[param]

                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[param]

                new_param = nn.Parameter(tensor.requires_grad_(True))
                group["params"][0] = new_param
                self.optimizer.state[new_param] = stored_state

                optimizable_tensors[group["name"]] = new_param
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        # self._colors = optimizable_tensors["color"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
                
        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        # self.denom = self.denom[valid_points_mask]
        # self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        self.target_mask = self.target_mask[valid_points_mask]
        
    def densification_postfix(self, new_xyz,new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "scaling" : new_scaling,
            "opacity": new_opacities,
            "rotation" : new_rotation,
            "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._colors = optimizable_tensors["color"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor[update_filter, 2:], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
    
    def densify_coarse(self, cam, data):
        # Densify regions in the rgb with fine details/frequencies based on target depth and rgb
        depth, rgb = data # 1,H,W : 3,H,W
        target_xyz = self._xyz[self.target_mask]
        
        # Get new xyz
        new_xyz, new_colors = ash(cam, target_xyz, depth.squeeze(0))
        new_rotation = torch.ones((new_xyz.shape[0],4)).cuda() * self._rotation[self.target_mask][0]
        random_index = torch.randint(0, self._scaling[self.target_mask].size(0), (1,)).item()
        new_scaling = torch.ones((new_xyz.shape[0],3)).cuda() * self._scaling[self.target_mask][random_index]
        new_opacity = torch.ones((new_xyz.shape[0],3)).cuda()
        new_opacity[:, 0] = torch.logit(new_opacity[:, 0])
        # Set w = 0.01 : As w_t = sig(w)*200, we need to set w = logit(w_t/200)
        new_opacity[:, 1] = (new_opacity[:, 1]*.01)
        # Finally set mu to 0 as the start of the traniing
        new_opacity[:, 2] = torch.logit(new_opacity[:, 2]*0.)
        
        new_target_mask = torch.ones((new_xyz.shape[0])).cuda().bool()
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)
        self.densification_postfix(new_xyz, new_colors,new_opacity, new_scaling, new_rotation)
        
    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        
        # self.split_spikes(extent, grads=grads, grad_threshold=max_grad)
        self.densify_and_clone(extent, grads=grads, grad_threshold=max_grad)
        self.densify_and_split(extent, grads=grads, grad_threshold=max_grad)
    
    def densify_and_clone(self, scene_extent, grads,grad_threshold ):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # print(torch.norm(grads, dim=-1).max())
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)


        # Only clone points in the scene not the target
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.target_mask)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_colors = self._colors[selected_pts_mask]
        
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask]
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)

        self.densification_postfix(new_xyz, new_colors,new_opacities, new_scaling, new_rotation)
        
    def dupelicate(self):
        selected_pts_mask = self.target_mask #torch.logical_and()
        new_xyz = self._xyz[selected_pts_mask] + torch.rand_like(self._xyz[selected_pts_mask])*0.005
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        # new_colors = self._colors[selected_pts_mask]
        
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask]
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_split(self, scene_extent, grads, grad_threshold):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        # Only select non-target points for densification
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.target_mask)
        
        N = 2
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_colors = self._colors[selected_pts_mask].repeat(N,1)
        
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask].repeat(N)
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)            

        self.densification_postfix(new_xyz,new_colors,new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def split_spikes(self,scene_extent, grads, grad_threshold, N=2):
                # Get scaling and split gaussians where S_max is > 10* S_min
        scaling = self.get_scaling
        selected_pts_mask = scaling.max()
        selected_pts_mask = (scaling.max(dim=1).values / scaling.min(dim=1).values.clamp(min=1e-8)) > 10
  
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.target_mask)
        
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask1 = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask1 = torch.logical_and(selected_pts_mask1,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        # Only select non-target points for densification
        selected_pts_mask = torch.logical_and(selected_pts_mask1, selected_pts_mask)
                
        new_rotation = self._rotation[selected_pts_mask]
        new_colors = self._colors[selected_pts_mask]

        # Update opacity
        new_opacity = self._opacity[selected_pts_mask]
        new_opacity[:, 0] = new_opacity[:, 0] *0.85
        
        # Update scaling
        scales = scaling[selected_pts_mask]
        _, max_indices = torch.max(scales, dim=1)  
        mask = torch.zeros_like(scales, dtype=torch.bool, device=scales.device)
        mask.scatter_(1, max_indices.unsqueeze(1), True)   
        scales[mask] = scales[mask] * 0.5
        scales[~mask] = scales[~mask] * 0.8 
        new_scaling = self.scaling_inverse_activation(scales)

        # Update position
        normals = rotated_soft_axis_direction(new_rotation, scaling[selected_pts_mask], type='max')
        normals = normals.contiguous().view(-1, 1, 3)
        cov3x3 = build_cov_matrix_torch(self.get_covariance())[selected_pts_mask]
        cov_inv = torch.linalg.inv(cov3x3) #.unsqueeze(1).repeat(1,3,1,1).view(-1, 3, 3)
        lam = torch.bmm(torch.bmm(normals, cov_inv), normals.transpose(1, 2)).squeeze(-1)  # (N,1,1)
        t = torch.sqrt(K_c / lam.clamp(min=1e-8))
        # print(normals.shape, t.shape, self._xyz[selected_pts_mask].shape)
        # exit()
        x1 = self._xyz[selected_pts_mask] + t * normals.squeeze(1)
        x2 = self._xyz[selected_pts_mask] - t * normals.squeeze(1)
        
        # Create the dupes
        new_xyz = torch.cat([x1, x2], dim=0)
        new_scaling = new_scaling.repeat(N,1)
        new_colors = new_colors.repeat(N,1)
        new_opacity = new_opacity.repeat(N,1)
        new_rotation = new_rotation.repeat(N,1)
        # print(new_colors.shape, new_opacity.shape, new_rotation.shape, new_scaling.shape, new_xyz.shape)
        # exit()
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask].repeat(N)
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)            

        self.densification_postfix(new_xyz,new_colors,new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def compute_topac_width(self, w, h, threshold=0.05):
        """ Implementing the function: t = m +- sqrt(-ln(0.05/h)/(w**2))
                        
            Notes:
                as this returns equi-distant values from m to the intersection with h, we can re-write the function as:
                    t = 2*(+sqrt(-ln(0.05/h)/(w**2)))
                    
                Actually to reduce computation, we calulcate just: torch.sqrt(-torch.log(threshold / h) / (w ** 2))
                The 2x factor apears in the width calculation as a half, meaning a tiny bit better
                        
        """
        return torch.sqrt(-torch.log(threshold / h) / (w ** 2))

    def prune(self,cam_list): # h_thresold, extent, max_screen_size,):
        """
        """
        h = self.get_coarse_opacity_with_3D_filter
        h_mask = (h < 0.05).squeeze(-1)
        
        prune_mask = torch.zeros_like(self.target_mask, dtype=torch.uint8).cuda()
        for cam in cam_list:
            prune_mask += get_in_view_dyn_mask(cam, self.get_xyz)

        prune_mask = prune_mask < len(cam_list)-2
        prune_mask = torch.logical_and(prune_mask, self.target_mask)

        prune_mask = torch.logical_or(prune_mask, h_mask)
        # if max_screen_size:
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

        #     prune_mask = torch.logical_or(prune_mask, big_points_vs)
        #     prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
          
    def reset_opacity(self):
        print('resetting opacity')
        opacities_new = self.get_opacity
        opacities_new[:,0] =  torch.logit(torch.tensor(0.05)).item()
        opacities_new[:,1] = 1.5

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight,
                           minview_weight, tvtotal1_weight, spsmoothness_weight, minmotion_weight):
        tvtotal = 0
        l1total = 0
        tstotal = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for index, grids in enumerate(self._deformation.deformation_net.grid.grids_()):
            if index in [0,1,3]: # space only
                for grid in grids:
                    tvtotal += compute_plane_smoothness(grid)
            elif index in [2,4,5]:
                for grid in grids: # space time
                    tstotal += compute_plane_smoothness(grid)
                    
                    l1total += torch.abs(1. - grid).mean()
            # else:
            #     for grid in grids: # space time
            #         l1total += torch.abs(1. - grid).mean()
        
        # Order is Yh0 Yl1 Yl0
        # tvtotal1 = 0
        # spsmoothness = 0
        # minmotion = 0
        # minview = 0
        # for index, grids in enumerate(self._deformation.deformation_net.grid.waveplanes_list()):
        #     if index in [0,1,3]: # space only
        #         if tvtotal1_weight > 0. or spsmoothness_weight
        #         for idx, grid in enumerate(grids):
        #             # Total variation
        #             if idx == 0:
        #                 tvtotal1 += compute_plane_tv(grid)
                    
        #             # Spatial smoothness prioritizing coarse features
        #             elif idx == 1:
        #                 spsmoothness += 0.6*grid.abs().mean()
        #             elif idx == 2:
        #                 spsmoothness += 0.3*grid.abs().mean()
                        
        #     elif index in [2,4,5]: # space time
        #         for idx, grid in enumerate(grids):
        #             # Min motion
        #             if idx == 0: 
        #                 minmotion += grid.abs().mean()
        #             elif idx == 1:
        #                 minmotion += 0.6 * grid.abs().mean()
        #             elif idx == 2:
        #                 minmotion += 0.3 * grid.abs().mean()
        #     else:
        #         for idx, grid in enumerate(grids):
        #             if idx == 0: 
        #                 minview += grid.abs().mean()
        #             elif idx == 1:
        #                 minview += 0.6 * grid.abs().mean()
        #             elif idx == 2:
        #                 minview += 0.3 * grid.abs().mean()
        
        return plane_tv_weight * tvtotal + time_smoothness_weight*tstotal + l1_time_planes_weight*l1total 
            # tvtotal1 * tvtotal1_weight + spsmoothness * spsmoothness_weight + minmotion * minmotion_weight + minview * minview_weight

    def compute_covariance_loss(self, xyz, rotation, scaling):
        # row, col = knn_graph(xyz, k=2, batch=None, loop=False)
        row, col  = self.target_neighbours

        cov = self.covariance_activation(scaling.detach(), 1, rotation.detach())
        
        A = xyz[row]
        B = xyz[col]
        
        t_AB = compute_alpha_interval(B - A, cov[row], alpha_threshold=0.1)
        t_BA = compute_alpha_interval(A-B, cov[col], alpha_threshold=0.1)

        # When this is true the Gaussians do not intersect
        # TODO: check if any negative values
        mask = t_AB + t_BA < 1.

        # print(((B[mask]-A[mask]).abs()).mean())
        return ((B[mask]-A[mask])**2).mean()

    def get_waveplanes(self):
        return self._deformation.deformation_net.grid.waveplanes_list()
    
    def update_wavelevel(self):
        self._deformation.update_wavelevel()
    
    def generate_neighbours(self, points):
        # Maybe we can get NN by sampling every 0.25 time steps
        # getting the neighbours at each time step? Maybe by selecting the closest
        # points independant of time?
        edge_index = knn_graph(points, k=5, batch=None, loop=False)
        self.target_neighbours = edge_index

    def update_neighbours(self, points):
        edge_index = knn_graph(points, k=5, batch=None, loop=False)
        self.target_neighbours = edge_index

    def compute_normals_rigidity(self, norms):
        points = self._xyz[self.target_mask] 
        row, col = self.target_neighbours
        disances = torch.norm(points[row] - points[col], dim=1)
        distance_weights = 1./disances.unsqueeze(-1).detach() # 1. - torch.exp(- (0.00018/ (disances**2)))
        diff = ((distance_weights*(norms[row] - norms[col]).abs())).mean()
        return diff
    
    def compute_displacement_rigidity(self, position):
        row, col = self.target_neighbours
        return torch.norm(position[row] - position[col], dim=1).unsqueeze(1)

    def compute_rigidity_loss(self, iteration):
        points = self._xyz[self.target_mask] 
        row, col = self.target_neighbours
    
        # As distance increased weighting decreases
        # distance_weights = 1. #/(torch.norm(points[row] - points[col], dim=1))
        # dx_coefs, drot, dcol, dcol_base = self.get_dyn_coefs
        
        # Dynamic rigidity w.r.t motion 
        #   To think about, maybe we take A,B,C,D as points where A and D
        # dx_coefs = dx_coefs.view(-1, 3, 3) # N, 3, 3
        
        # Smooth loss
        # diff = ((distance_weights*(dx_coefs[row] - dx_coefs[col])**2).mean(-1).mean(-1)).mean()
        
        # Regularize w w.r.t nn
        w = self.get_wopac
        # 1-\ \exp\left(-\frac{0.0009}{\left(2\cdot x^{2}\right)}\right)
        disances = torch.norm(points[row] - points[col], dim=1)
        distance_weights = 1.#/disances # 1. - torch.exp(- (0.00018/ (disances**2)))
        diff = ((distance_weights*(w[row] - w[col]).abs())).mean()
        #((distance_mask*(((dx_coefs[row] - dx_coefs[col])**2).mean(-1)))).mean()
        # Get the rotation of the point at t = 0 and t = 1
        # rot0 = self.rotation_activation(self._rotation[self.target_mask])
        # rot1 = self.rotation_activation(self._rotation[self.target_mask] + drot[:, -1, :])
        # # Get the normalized direction of the smallest axis of each point
        # #  TODO: We should 
        # norms0 = rotated_soft_axis_direction(rot0,self.get_scaling[self.target_mask])
        # # @ t = 1
        # norms1 = rotated_soft_axis_direction(rot1,self.get_scaling[self.target_mask])

        # # angles between pairs of points
        # dot_A = torch.sum(norms0[row] * norms0[col], dim=1)  # (E,)
        # dot_B = torch.sum(norms1[row] * norms1[col], dim=1)  # (E,)

        # # The goal is to keep relative angles (dot products) consistent
        # diff += F.mse_loss(dot_A, dot_B)

        return diff
    
    def compute_static_rigidity_loss(self, iteration):
        points = self._xyz[self.target_mask]
        row, col = self.target_neighbours
        
        distance_mask = 0.1 > torch.norm(points[row] - points[col], dim=1)  # (E,)
        
        rot = self.rotation_activation(self._rotation[self.target_mask])
        norms1 = rotated_soft_axis_direction(rot, self.get_scaling[self.target_mask])
        scales = self.get_scaling.min(dim=1).values

        # Surface direction, surface thickness
        diff = ((distance_mask*(((norms1[row] - norms1[col])**2).mean(-1)))).mean()
        diff += ((distance_mask*(((scales[row] - scales[col])**2).mean(-1)))).mean()

        return diff

def compute_alpha_interval(d, cov6, alpha_threshold=0.1):
    """
    d: (N, 3) - vector from A to B
    cov6: (N, 6) - compact covariance for point A
    """
    # Step 1: Unpack 6D compact covariances into full 3x3 matrices
    N = cov6.shape[0]
    cov = torch.zeros((N, 3, 3), device=cov6.device, dtype=cov6.dtype)
    cov[:, 0, 0] = cov6[:, 0]
    cov[:, 1, 1] = cov6[:, 1]
    cov[:, 2, 2] = cov6[:, 2]
    cov[:, 0, 1] = cov[:, 1, 0] = cov6[:, 3]
    cov[:, 0, 2] = cov[:, 2, 0] = cov6[:, 4]
    cov[:, 1, 2] = cov[:, 2, 1] = cov6[:, 5]

    # Add small regularization for stability
    # eps = 1e-6
    # cov[:, range(3), range(3)] += eps
    try:
        cov_inv = torch.linalg.inv(cov)
    except RuntimeError:
        raise ValueError("Covariance matrix is singular or ill-conditioned.")

    # Step 2: Compute λ = d^T @ Σ⁻¹ @ d (Mahalanobis squared along direction d)
    d_exp = d.unsqueeze(1)  # (N, 1, 3)
    λ = torch.bmm(torch.bmm(d_exp, cov_inv), d_exp.transpose(1, 2)).squeeze(-1).squeeze(-1)  # (N,)
    λ = λ.clamp(min=1e-10)
    
    # Step 3: Compute cutoff t where Gaussian drops below alpha_threshold
    c = -2.0 * torch.log(torch.tensor(alpha_threshold, device=d.device, dtype=d.dtype))
    t_cutoff = torch.sqrt(c / λ)  # (N,)

    return t_cutoff  # alpha < threshold when |t| > t_cutoff

K_c = -2*torch.log(torch.tensor(0.6)).cuda()
import torch.nn.functional as F
def quaternion_rotate(q, v):
    q_vec = q[:, :3]
    q_w = q[:, 3].unsqueeze(1)
    t = 2 * torch.cross(q_vec, v, dim=1)
    return v + q_w * t + torch.cross(q_vec, t, dim=1)

def rotated_soft_axis_direction(r, s, temperature=10.0, type='min'):
    # s: (N, 3), we want the direction of the smallest abs scale
    abs_s = torch.abs(s)

    # Step 1: Compute softmin weights (lower abs(s) => higher weight)
    if type == 'min':
        weights = F.softmax(-abs_s * temperature, dim=1)  # (N, 3)
    elif type == 'max':
        weights = F.softmax(abs_s * temperature, dim=1)  # (N, 3)

    # Step 2: Basis axes: x, y, z
    basis = torch.eye(3, device=s.device).unsqueeze(0)  # (1, 3, 3)

    # Step 3: Weighted sum of basis vectors
    soft_axis = torch.bmm(weights.unsqueeze(1), basis.repeat(s.size(0), 1, 1)).squeeze(1)  # (N, 3)

    # Step 4: Rotate the direction
    rotated = quaternion_rotate(r, soft_axis)  # (N, 3)

    return rotated

import torch.nn.functional as F
def min_pool_nonzero(depth_map, patch_size):
    """
    Computes patch-wise minimum non-zero depth, then upsamples back to original size.

    Args:
        depth_map (Tensor): [H, W], depth values with 0 = missing
        patch_size (int): size of square patches

    Returns:
        Tensor: [H, W] with min depth per patch, upsampled to original size
    """
    H, W = depth_map.shape

    # Replace zeros with large value so they don't interfere with min
    masked = depth_map.clone()
    masked[masked == 0] = float('inf')

    # Trim to be divisible by patch size
    H_trim, W_trim = H - H % patch_size, W - W % patch_size
    masked = masked[:H_trim, :W_trim]

    # Reshape into patches
    reshaped = masked.view(H_trim // patch_size, patch_size, W_trim // patch_size, patch_size)
    patches = reshaped.permute(0, 2, 1, 3).reshape(H_trim // patch_size, W_trim // patch_size, -1)

    # Min per patch
    patch_min, _ = patches.min(dim=-1)
    patch_min[patch_min == float('inf')] = 0  # Restore 0 for all-zero patches

    # Upsample to original resolution
    patch_min = patch_min.unsqueeze(0).unsqueeze(0)  # [1, 1, H', W']
    upsampled = F.interpolate(patch_min, size=(H, W), mode='nearest')
    
    return upsampled.squeeze(0).squeeze(0) 

def backproject_depth_to_xyz(depth_map, camera):
    H, W = depth_map.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=depth_map.device),
        torch.arange(W, device=depth_map.device),
        indexing='ij'
    )
    
    fx = fy = 0.5 * H / np.tan(camera.FoVy / 2)  # estimated from FOV and image size
    fx  = 0.5 * W / np.tan(camera.FoVx / 2)  # estimated from FOV and image size
    cx = camera.image_width / 2
    cy = camera.image_height / 2

    z = depth_map
    x3d = (x - cx) * z / fx
    y3d = (y - cy) * z / fy
    xyz = torch.stack([x3d, y3d, z], dim=-1)  # [H, W, 3]

    valid = z > 0
    return xyz[valid], y[valid], x[valid]

def populate_background(camera, xyz, col) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    # Visibility check
    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2].abs() <= 1)
    visible_mask = in_ndc_bounds & in_front
    
    # Compute pixel coordinates
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long()
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long()    # Init mask values

    # Only sample pixels for visible points
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]
    px_valid = px[valid_idx].clamp(0, camera.image_width - 1)
    py_valid = py[valid_idx].clamp(0, camera.image_height - 1)
    
    # Get mask
    mask = camera.mask.to(device)  # (H, W)
    sampled_mask = mask[py_valid, px_valid] 
    
    mask_valid = sampled_mask.bool()
    final_idx = valid_idx[mask_valid]
    img = torch.zeros_like(mask).cuda()
    img[py_valid, px_valid]  = proj_xyz[valid_idx,3]
    pcd_mask = min_pool_nonzero(img, 50) > 0. # Get a mask where 1 includes dilater regions to not sample new pcds
    uniform_depth = torch.zeros_like(pcd_mask).cuda() # H,W
    stride = 25
    uniform_depth[::stride, ::stride] = proj_xyz[valid_idx,3].max() # Set the max depth w.r.t camera
    uniform_depth[pcd_mask] = 0. # Apply the blur mask to avoid selecting samples within the field
    
    # Reproject local points into global space
    py, px = torch.nonzero(uniform_depth > 0, as_tuple=True)
    depths = uniform_depth[py, px]
    
    x_ndc = (px.float() / camera.image_width) * 2 - 1
    y_ndc = (py.float() / camera.image_height) * 2 - 1
    clip_coords = torch.stack([x_ndc * depths, y_ndc * depths, depths, depths], dim=-1)  # (N, 4)
    
    world_coords_h = clip_coords @ torch.inverse(camera.full_proj_transform.to(device)).T  # (N, 4)
    world_coords = world_coords_h[:, :3] / world_coords_h[:, 3:4]  # convert to 3D
    
    xyz_new = torch.cat([xyz, world_coords], dim=0)
    col_new = torch.cat([col, torch.rand(world_coords.shape[0], 3).cuda()], dim=0)
    return xyz_new, col_new
    # img = min_pool_nonzero(img, 25)
    import matplotlib.pyplot as plt
    
    imgs = [img, pcd_mask, uniform_depth,]  # Replace these with your actual image tensors

    fig, axes = plt.subplots(1, len(imgs), figsize=(8, 8))  # 2 rows, 2 columns
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img.cpu(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    exit()

    plt.imshow(mask.cpu(), cmap='hot')
    plt.title("Local Variation (RGB)")
    plt.colorbar()
    plt.show()
    exit()
    return mask_values.long()

def get_in_view_dyn_mask(camera, xyz) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    # Visibility check
    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2].abs() <= 1)
    visible_mask = in_ndc_bounds & in_front
    
    # Compute pixel coordinates
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long()
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long()    # Init mask values

    # Only sample pixels for visible points
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]
    px_valid = px[valid_idx].clamp(0, camera.image_width - 1)
    py_valid = py[valid_idx].clamp(0, camera.image_height - 1)
    
    mask = camera.mask.to(device)  # (H, W)
    sampled_mask = mask[py_valid, px_valid].bool()
    # Get filtered 3D points and colors
    final_mask = torch.zeros(N, dtype=torch.uint8, device=device)
    final_mask[valid_idx[sampled_mask]] = 1  # Set mask to 1 where points are visible and inside the mask

    return final_mask 
    
    img = camera.original_image.permute(1,2,0).cuda()

    depth_img = torch.zeros((camera.image_height, camera.image_width), device=device)
    depth_img[py_valid[mask_valid], px_valid[mask_valid]] = proj_z
     
    # Take minimum distance w.r.t patch (avoid placing background)
    # depth_img = min_pool_nonzero(depth_img, 15) # * mask - multiplication is pointsless as the variance is already mask
    # import matplotlib.pyplot as plt
    # tensor_hw = depth_img.cpu()  # If it's on GPU
    # plt.imshow(tensor_hw, cmap='gray')
    # plt.axis('off')
    # plt.show()

    img = img.permute(2,0,1) * mask
    kernel_size=27
    C=3
    padding = kernel_size // 2
    weight = torch.ones((C, 1, kernel_size, kernel_size), dtype=img.dtype, device=img.device)
    weight /= kernel_size * kernel_size

    mean = F.conv2d(img, weight, padding=padding, groups=C) # local mean
    mean_of_squares = F.conv2d(img ** 2, weight, padding=padding, groups=C) # local mean of squares 
    variance = mean_of_squares - mean ** 2 # Variance E[x^2] - (E[x])^2

    # Mask variance and normalize it
    variance = variance * mask
    variance = (variance - variance.min())/(variance.max() - variance.min())
    import matplotlib.pyplot as plt
    tensor_hw = variance.permute(1,2,0).mean(-1).cpu() > 0.001  # If it's on GPU
    plt.imshow(tensor_hw, cmap='gray')
    plt.axis('off')
    plt.show()
    
    stride = 5
    mask = torch.zeros_like(variance).cuda()
    mask[::stride, ::stride] = variance[::stride, ::stride]
    
    new_xyz = (mask > 0.001)
    depths_ = torch.where(new_xyz, depth_img, 0.)
    depths = torch.zeros_like(depth_img).cuda()
    depths[new_xyz] = depths_[new_xyz]
    
    new_xyz_from_depth, py_valid, px_valid = backproject_depth_to_xyz(depths, camera)

    # Step 2: Get RGB at those (py, px)
    # Shape of original image: [3, H, W]
    img = camera.original_image.cuda()
    rgb = img[:, py_valid, px_valid].permute(1, 0) 
    
    # Concatenate
    final_xyz = torch.cat([xyz_in_mask, new_xyz_from_depth], dim=0)  # [N+M, 3]
    final_rgb = torch.cat([col_in_mask, rgb], dim=0)  # [N+M, 3]
    return final_xyz, final_rgb
    import matplotlib.pyplot as plt

    tensor_hw = depths.cpu()  # If it's on GPU
    plt.imshow(tensor_hw, cmap='gray')
    plt.axis('off')
    plt.show()
    exit()
    
    import matplotlib.pyplot as plt
    plt.imshow(mask.cpu(), cmap='hot')
    plt.title("Local Variation (RGB)")
    plt.colorbar()
    plt.show()
    exit()
    return mask_values.long()

def ash(camera, xyz, depth) -> torch.Tensor:
    camera = camera[0]
    device = xyz.device
    N = xyz.shape[0]

    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    # Visibility check
    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2].abs() <= 1)
    visible_mask = in_ndc_bounds & in_front
    
    # Compute pixel coordinates
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long()
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long()    # Init mask values

    # Only sample pixels for visible points
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]
    px_valid = px[valid_idx].clamp(0, camera.image_width - 1)
    py_valid = py[valid_idx].clamp(0, camera.image_height - 1)
    
    mask = camera.mask.to(device)  # (H, W)
    sampled_mask = mask[py_valid, px_valid] 
    
    mask_valid = sampled_mask.bool()
    final_idx = valid_idx[mask_valid]

    # Get filtered 3D points and colors
    xyz_in_mask = xyz[final_idx]
    proj_z = proj_xyz[final_idx, 2]
    # return xyz_in_mask, col_in_mask
    
    img = camera.original_image.permute(1,2,0).cuda()


    # tensor_hw = depth_img.cpu()  # If it's on GPU
    # plt.imshow(tensor_hw, cmap='gray')
    # plt.axis('off')
    # plt.show()

    img = img.permute(2,0,1) * mask
    kernel_size=5
    C=3
    padding = kernel_size // 2
    weight = torch.ones((C, 1, kernel_size, kernel_size), dtype=img.dtype, device=img.device)
    weight /= kernel_size * kernel_size

    mean = F.conv2d(img, weight, padding=padding, groups=C) # local mean
    mean_of_squares = F.conv2d(img ** 2, weight, padding=padding, groups=C) # local mean of squares 
    variance = mean_of_squares - mean ** 2 # Variance E[x^2] - (E[x])^2

    # Mask variance and normalize it
    variance = variance.mean(0) * mask
    variance = (variance - variance.min())/(variance.max() - variance.min())
    
    stride = 5
    mask = torch.zeros_like(variance).cuda()
    mask[::stride, ::stride] = variance[::stride, ::stride]
    
    new_xyz = (mask > 0.0001)
    print('sum:', new_xyz.sum())
    depth_img = depth
    depths_ = torch.where(new_xyz, depth_img, 0.)
    depths = torch.zeros_like(depth_img).cuda()
    depths[new_xyz] = depths_[new_xyz]
    
    new_xyz_from_depth, py_valid, px_valid = backproject_depth_to_xyz(depths, camera)

    # Step 2: Get RGB at those (py, px)
    # Shape of original image: [3, H, W]
    img = camera.original_image.cuda()
    rgb = img[:, py_valid, px_valid].permute(1, 0) 

    return new_xyz_from_depth, rgb
    import matplotlib.pyplot as plt

    tensor_hw = (depth_img)
    tensor_hw[camera.mask.to(device) > 0] *= .5
    plt.imshow(new_xyz.cpu(), cmap='gray')
    plt.axis('off')
    plt.show()
    exit()
    
    import matplotlib.pyplot as plt
    plt.imshow(mask.cpu(), cmap='hot')
    plt.title("Local Variation (RGB)")
    plt.colorbar()
    plt.show()
    exit()
    return mask_values.long()



def refilter_pcd(camera, xyz, col) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    # Visibility check
    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2].abs() <= 1)
    visible_mask = in_ndc_bounds & in_front
    
    # Compute pixel coordinates
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long()
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long()    # Init mask values

    # Only sample pixels for visible points
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]
    px_valid = px[valid_idx].clamp(0, camera.image_width - 1)
    py_valid = py[valid_idx].clamp(0, camera.image_height - 1)
    
    mask = camera.mask.to(device)  # (H, W)
    sampled_mask = mask[py_valid, px_valid] 
    
    mask_valid = sampled_mask.bool()
    final_idx = valid_idx[mask_valid]

    # Get filtered 3D points and colors
    xyz_in_mask = xyz[final_idx]
    col_in_mask = col[final_idx]
    proj_z = proj_xyz[final_idx, 2]
    return xyz_in_mask, col_in_mask

def build_cov_matrix_torch(cov6):
    N = cov6.shape[0]
    cov = torch.zeros((N, 3, 3), device=cov6.device, dtype=cov6.dtype)
    cov[:, 0, 0] = cov6[:, 0]  # σ_xx
    cov[:, 0, 1] = cov[:, 1, 0] = cov6[:, 1]  # σ_xy
    cov[:, 0, 2] = cov[:, 2, 0] = cov6[:, 2]  # σ_xz
    cov[:, 1, 1] = cov6[:, 3]  # σ_yy
    cov[:, 1, 2] = cov[:, 2, 1] = cov6[:, 4]  # σ_yz
    cov[:, 2, 2] = cov6[:, 5]  # σ_zz
    return cov

def get_in_view_screenspace(camera, xyz: torch.Tensor) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # Apply full projection (world → clip space)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    # Homogeneous divide to get NDC coordinates
    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    # Check if points are in front of the camera and within [-1, 1] in all 3 axes
    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2].abs() <= 1)

    # Final visibility mask (points that would fall within the image bounds)
    visible_mask = in_front & in_ndc_bounds

    return visible_mask.long()

SQRT_PI = torch.sqrt(torch.tensor(torch.pi))
def gaussian_integral(w):
    """Returns high weight (0 to 1) for solid materials
    
        Notes on optimization:
            The initial integral for a gaussian from 0 to 1 is (SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2), 
            with the center at 0. Instead, to be more precise/sensitive and faster we evaluate the integral between
            -1 and 1 with the gaussian centered at mu=0, as SQRT_PI*erf1/w. This reduces the complexity of
            the integral
    """
    SQRT_PI = torch.sqrt(torch.tensor(torch.pi, dtype=w.dtype, device=w.device))
    EPS = 1e-8  # for numerical stability
    return (SQRT_PI / (w + EPS)) * torch.erf(w)

import random
def get_sorted_random_pair():
    """Get a pair of random floats betwen 0 and 1
    """
    num1 = random.random()
    num2 = random.random()
    return num1, num2

# from scipy.spatial import KDTree
# def distCUDA2(points):
#     points_np = points.detach().cpu().float().numpy()
#     dists, inds = KDTree(points_np).query(points_np, k=4)
#     meanDists = (dists[:, 1:] ** 2).mean(1)
#     return torch.tensor(meanDists, dtype=points.dtype, device=points.device)