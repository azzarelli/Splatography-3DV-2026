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
# from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness,compute_plane_tv

from gaussian_renderer import render_motion_point_mask


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

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._xyz_background = torch.empty(0)
        
        self._opacityh = torch.empty(0)
        self._opacityw = torch.empty(0)
        self._opacitymu = torch.empty(0)

        self._deformation = deform_network(args)
        
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.spatial_lr_scale_background = 0
        self.target_mask = None
        self.target_neighbours = None
        
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._xyz_background,
            self._deformation.state_dict(),
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacityh,
            self._opacityw,
            self._opacitymu,
            self.filter_3D,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.spatial_lr_scale_background,
            self.target_mask
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._xyz_background,
        deform_state,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacityh,
        self._opacityw,
        self._opacitymu,
        self.filter_3D,
        opt_dict,
        self.spatial_lr_scale,self.spatial_lr_scale_background, self.target_mask) = model_args

        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)

        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
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
        return self._xyz, self._xyz_background

    @property
    def get_opacity(self):
        return self._opacityh, self._opacityw, self._opacitymu
    
    @property
    def get_coarse_opacity_with_3D_filter(self):
        opacity = torch.sigmoid(self._opacityh)
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
        return torch.sigmoid(self._opacityh)
    @property
    def get_wopac(self):
        return self._opacityw
    @property
    def get_muopac(self):
        return torch.sigmoid(self._opacitymu)
    

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        #TODO consider focal length and image width
        xyz = torch.cat([self._xyz, self._xyz_background], dim=0)
        
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
        
    def create_from_pcd(self, pcd : BasicPointCloud, cam_list=None, dataset_type="dynerf"):
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        # Construct pcd, increase number of points in target region
        # increase scene points as well
        # Remove unseen points
        # Remove points betwen 1 and  target view masks

        if dataset_type == "dynerf": # This is for the dynerf dataset
            dyn_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()
            scene_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()
            
            for cam in cam_list:             
                dyn_mask += get_in_view_dyn_mask(cam, fused_point_cloud).long()
                scene_mask += get_in_view_screenspace(cam, fused_point_cloud).long()
            
            scene_mask = scene_mask > 0
            target_mask = dyn_mask > (len(cam_list)-1)
            dyn_mask = target_mask # torch.logical_or(target_mask, dyn_mask == 0)
            viable  = torch.logical_and(dyn_mask, scene_mask)
            
        elif dataset_type == "condense":
            target_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()
            # Pre-defined corners from the ViVo dataset (theres one for each scene butre-using the same one doesnt cause problems)
            CORNERS = [[-1.38048, -0.1863],[-0.7779, 1.6705], [1.1469, 1.1790], [0.5832, -0.7245]]
            polygon = np.array(CORNERS)  # shape (4, 2)
            from matplotlib.path import Path
            path = Path(polygon)
            points_xy = fused_point_cloud[:, 1:].cpu().numpy()  # (N, 2)
            # Create mask for points inside polygon
            viable = torch.from_numpy(path.contains_points(points_xy)).cuda()
            
            
        # Downsample background gaussians
        pcds = fused_point_cloud[~viable].cpu().numpy().astype(np.float64)
        cols = fused_color[~viable].cpu().numpy().astype(np.float64)
        
        # Re-sample point cloud
        target = fused_point_cloud[viable]
        target_col = fused_color[viable]

        if dataset_type == "condense":
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcds)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            # Voxel size controls the granularity
            voxel_size = 0.05  # Adjust based on your data scale
            downsampled_pcd = pcd.voxel_down_sample(voxel_size)

            # Convert back to PyTorch tensor
            bck_pcds = torch.tensor(np.asarray(downsampled_pcd.points), dtype=fused_point_cloud.dtype).cuda()
            bck_cols = torch.tensor(np.asarray(downsampled_pcd.colors), dtype=fused_color.dtype).cuda()
            
            # pcds = fused_point_cloud[viable].cpu().numpy().astype(np.float64)
            # cols = fused_color[viable].cpu().numpy().astype(np.float64)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcds)
            # pcd.colors = o3d.utility.Vector3dVector(cols)

            # # Voxel size controls the granularity
            # voxel_size = 0.03  # Adjust based on your data scale
            # downsampled_pcd = pcd.voxel_down_sample(voxel_size)

            # # Convert back to PyTorch tensor
            # target = torch.tensor(np.asarray(downsampled_pcd.points), dtype=fused_point_cloud.dtype).cuda()
            # target_col = torch.tensor(np.asarray(downsampled_pcd.colors), dtype=fused_color.dtype).cuda()
            pcds = bck_pcds
            cols = bck_cols
            
        else:
            pcds = torch.tensor(pcds, dtype=fused_point_cloud.dtype).cuda()
            cols = torch.tensor(cols, dtype=fused_color.dtype).cuda()
            
            mask = None
            for cam in cam_list:
                x = torch.tensor(cam.T).cuda().unsqueeze(0)
                temp = torch.norm(x - pcds, dim=-1) < 51.
                if mask == None:
                    mask = temp
                else:
                    mask = mask & temp
                                 
            pcds = pcds[mask, :]
            cols = cols[mask, :]
            

        
            
        fused_point_cloud = torch.cat([pcds, target], dim=0)
        fused_color = torch.cat([cols, target_col], dim=0)
        target_mask = torch.zeros((fused_color.shape[0], 1)).cuda()
        target_mask[cols.shape[0]:, :] = 1
        target_mask = (target_mask > 0.).squeeze(-1)
    
        if dataset_type == "dynerf":
            while target_mask.sum() < 30000:
                target_point_noise =  fused_point_cloud[target_mask] + torch.randn_like(fused_point_cloud[target_mask]).cuda() * 0.05
                fused_point_cloud = torch.cat([fused_point_cloud,target_point_noise], dim=0)
                fused_color = torch.cat([fused_color,fused_color[target_mask]], dim=0)
                target_mask = torch.cat([target_mask, target_mask[target_mask]])
            
            while (~target_mask).sum() < 60000:
                target_point_noise =  fused_point_cloud[~target_mask] + torch.randn_like(fused_point_cloud[~target_mask]).cuda() * 0.1
                fused_point_cloud = torch.cat([fused_point_cloud,target_point_noise], dim=0)
                fused_color = torch.cat([fused_color,fused_color[~target_mask]], dim=0)
                target_mask = torch.cat([target_mask, target_mask[~target_mask]])

        self.target_mask = target_mask
        # print(self.target_mask.sum(), self.target_mask.shape)
        # exit()
        # Prune background down to 100k
        if dataset_type == "condense":
            err = 0.05
        else:
            err = 0.1
        xyz_min = fused_point_cloud[target_mask].min(0).values - err
        xyz_max = fused_point_cloud[target_mask].max(0).values + err
        self._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
        
        xyz_min = fused_point_cloud[~target_mask].min(0).values
        xyz_max = fused_point_cloud[~target_mask].max(0).values
        self._deformation.deformation_net.set_aabb(xyz_max, xyz_min, grid_type='background')
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # fused_color[~target_mask] = fused_color[~target_mask] + torch.clamp(torch.rand(fused_color[~target_mask].shape[0], 3).cuda()*0.1, 0., 1.)
        
        if dataset_type == "condense":
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud[target_mask]), 0.00000000001)
            dist2_else = torch.clamp_min(distCUDA2(fused_point_cloud[~target_mask]), 0.00000000001)
            dist2 = torch.cat([dist2_else, dist2], dim=0)
        else:
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)

        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Initialize opacities
        opacities = 1. * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        if dataset_type == "condense":
            # Set h = 1 : As max_opac = sig(h) to set max opac = 1 we need h = logit(1)
            opacities[:, 0] = torch.logit(opacities[:, 0]*0.95)
            # Set w = 0.01 : As w_t = sig(w)*200, we need to set w = logit(w_t/200)
            opacities[:, 1] = (opacities[:, 1]*1.5)
            # Finally set mu to 0 as the start of the traniing
            opacities[:, 2] = torch.logit(opacities[:, 2]*0.5)
        else:
            opacities[:, 0] = torch.logit(opacities[:, 0]*0.95)
            opacities[:, 1] = (opacities[:, 1]*1.5)
            opacities[:, 2] = torch.logit(opacities[:, 2]*0.5)
        
        self._xyz = nn.Parameter(fused_point_cloud[target_mask].requires_grad_(True))
        self._xyz_background = nn.Parameter(fused_point_cloud[~target_mask].requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        self._deformation = self._deformation.to("cuda")
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacityh  = nn.Parameter(opacities[:, 0].unsqueeze(-1).requires_grad_(True))
        self._opacityw  = nn.Parameter(opacities[:, 1].unsqueeze(-1).requires_grad_(True))
        self._opacitymu = nn.Parameter(opacities[:, 2].unsqueeze(-1).requires_grad_(True))

        mean_foreground = fused_point_cloud[target_mask].mean(dim=0).unsqueeze(0)
        dist_foreground = torch.norm(fused_point_cloud[target_mask] - mean_foreground, dim=1)
        self.spatial_lr_scale = torch.max(dist_foreground).detach().cpu().numpy()
        
        mean_foreground = fused_point_cloud[~target_mask].mean(dim=0).unsqueeze(0)
        dist_foreground = torch.norm(fused_point_cloud[~target_mask] - mean_foreground, dim=1)
        self.spatial_lr_scale_background = torch.max(dist_foreground).detach().cpu().numpy()

        if dataset_type == "dynerf": # For the dynerf scene the background lr scale is too large
            self.spatial_lr_scale_background *= 0.1
        print(f"Target lr scale: {self.spatial_lr_scale} | Background lr scale: {self.spatial_lr_scale_background}")
        self.active_sh_degree = 0
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
    
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._xyz_background], 'lr': training_args.position_lr_init * self.spatial_lr_scale_background, "name": "xyz_background"},
            
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            
            {'params': list(self._deformation.get_background_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale_background, "name": "deformation_background"},
            {'params': list(self._deformation.get_background_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale_background, "name": "grid_background"},
            
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            
            {'params': [self._opacityh], 'lr': training_args.opacityh_lr, "name": "opacityh"},            
            {'params': [self._opacityw], 'lr': training_args.opacityw_lr, "name": "opacityw"},            
            {'params': [self._opacitymu], 'lr': training_args.opacitymu_lr, "name": "opacitymu"},            

            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.xyz_background_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale_background,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale_background,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.deformation_background_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale_background,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale_background,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.grid_background_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale_background,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale_background,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "xyz_background":
                lr = self.xyz_background_scheduler_args(iteration)
                param_group['lr'] = lr

            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                
            elif  "grid_background" in param_group["name"]:
                lr = self.grid_background_scheduler_args(iteration)
                param_group['lr'] = lr
                
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                
            elif param_group["name"] == "deformation_background":
                lr = self.deformation_background_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        
        for i in range(self._opacityh.shape[1]):
            l.append('opacityh_{}'.format(i))
        for i in range(self._opacityw.shape[1]):
            l.append('opacityw_{}'.format(i))
        for i in range(self._opacitymu.shape[1]):
            l.append('opacitymu_{}'.format(i))
        
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l
    
    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        
    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = torch.cat([self._xyz, self._xyz_background], dim=0).detach().cpu().numpy()
        
        opacityh = self._opacityh.detach().cpu().numpy()
        opacityw = self._opacityw.detach().cpu().numpy()
        opacitymu = self._opacitymu.detach().cpu().numpy()
        
        normals = np.zeros_like(xyz)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacityh,opacityw, opacitymu, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals, colors, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opach_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacityh")]
        opach_names = sorted(opach_names, key = lambda x: int(x.split('_')[-1]))
        opacityh = np.zeros((xyz.shape[0], len(opach_names)))
        for idx, attr_name in enumerate(opach_names):
            opacityh[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        opacw_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacityw")]
        opacw_names = sorted(opacw_names, key = lambda x: int(x.split('_')[-1]))
        opacityw = np.zeros((xyz.shape[0], len(opacw_names)))
        for idx, attr_name in enumerate(opacw_names):
            opacityw[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        opacmu_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacitymu")]
        opacmu_names = sorted(opacmu_names, key = lambda x: int(x.split('_')[-1]))
        opacitmu = np.zeros((xyz.shape[0], len(opacmu_names)))
        for idx, attr_name in enumerate(opacmu_names):
            opacitmu[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
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

        # TODO: check we have laoded target mask first (we should have)
        self._xyz = nn.Parameter(torch.tensor(xyz[self.target_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._xyz_background = nn.Parameter(torch.tensor(xyz[~self.target_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacityh = nn.Parameter(torch.tensor(opacityh, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacityw = nn.Parameter(torch.tensor(opacityw, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacitymu = nn.Parameter(torch.tensor(opacitmu, dtype=torch.float, device="cuda").requires_grad_(True))
        
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
            if len(group["params"])>1 or group["name"] == 'xyz_background':continue
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


    def densification_postfix(self, new_xyz,new_features_dc, new_features_rest, new_opacitiesh,
                              new_opacitiesw,new_opacitiesmu, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "scaling" : new_scaling,
            "opacityh": new_opacitiesh,
            "opacityw": new_opacitiesw,
            "opacitymu": new_opacitiesmu,
            "rotation" : new_rotation,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacityh = optimizable_tensors["opacityh"]
        self._opacityw = optimizable_tensors["opacityw"]
        self._opacitymu = optimizable_tensors["opacitymu"]
        
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
    
    def dupelicate(self):
        selected_pts_mask = self.target_mask #torch.logical_and()
        new_xyz = self._xyz+ torch.rand_like(self._xyz)*0.005
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacitiesh = self._opacityh[selected_pts_mask]
        new_opacitiesw = self._opacityw[selected_pts_mask]
        new_opacitiesmu = self._opacitymu[selected_pts_mask]
        
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask]
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacitiesh, new_opacitiesw,
                                   new_opacitiesmu, new_scaling, new_rotation)

    @torch.no_grad()
    def dynamic_dupelication(self):
        """Duplicate points with highly dynamic motion - maybe the top 10% of points with the largest motions?
        """
        
        selected_pts_mask = render_motion_point_mask(self)
        
        new_xyz = self._xyz[selected_pts_mask] + torch.rand_like(self._xyz[selected_pts_mask])*0.005
        
        target = self.target_mask.clone()
        target[target == True] = selected_pts_mask.clone()

        
        new_features_dc = self._features_dc[target]
        new_features_rest = self._features_rest[target]
        new_scaling = self._scaling[target]
        new_rotation = self._rotation[target]
        new_opacitiesh = self._opacityh[target]
        new_opacitiesw = self._opacityw[target]
        new_opacitiesmu = self._opacitymu[target]
        
        # Update target mask
        new_target_mask = self.target_mask[target]
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacitiesh,new_opacitiesw,new_opacitiesmu, new_scaling, new_rotation)
          
    def reset_opacity(self):
        print('resetting opacity')
        opacities_newh = self._opacityh
        # opacities_neww = self._opacityw
        opacities_newh =  torch.logit(torch.tensor(0.05)).item() + opacities_newh * 0.
        # opacities_neww = 1.5 + opacities_neww

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_newh, "opacityh")
        self._opacityh = optimizable_tensors["opacityh"]
    
        # optimizable_tensors = self.replace_tensor_to_optimizer(opacities_neww, "opacityw")
        # self._opacityw = optimizable_tensors["opacityw"]

    def compute_regulation(self, time_smoothness_weight, plane_tv_weight,
                           l1_time_planes_weight,l1_col_planes_weight, 
                           tv_background_weight, ts_background_weight):
        tvtotal = 0
        l1total = 0
        tstotal = 0
        col=0
        
        tvbackground = 0
        tsbackground = 0
        
        wavelets = self._deformation.deformation_net.grid.waveplanes_list()
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for index, grids in enumerate(self._deformation.deformation_net.grid.grids_()):
            if index in [0,1,3]: # space only
                for grid in grids:
                    tvtotal += compute_plane_smoothness(grid)
            elif index in [2, 4, 5]:
                for grid in grids: # space time
                    tstotal += compute_plane_smoothness(grid)
                
                for grid in wavelets[index]:
                    l1total += torch.abs(grid).mean()
                    
            elif index in [6, 7, 8]:
                for grid in wavelets[index]: # space time
                    col += torch.abs(grid).mean()
                    
        wavelets = self._deformation.deformation_net.background_grid.waveplanes_list()
        # # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for index, grids in enumerate(self._deformation.deformation_net.background_grid.grids_()):
            if index in [0,1,3]: # space only
                for grid in grids:
                    tvbackground += compute_plane_smoothness(grid)
            elif index in [2, 4, 5]:
                for grid in grids: # space time
                    tsbackground += compute_plane_smoothness(grid)       
        
        return plane_tv_weight * tvtotal + time_smoothness_weight*tstotal + l1_time_planes_weight*l1total + l1_col_planes_weight*col +\
            ts_background_weight*tsbackground + tv_background_weight*tvbackground



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

    # exit()
    # Get filtered 3D points and colors
    final_mask = torch.zeros(N, dtype=torch.uint8, device=device)
    final_mask[valid_idx[sampled_mask]] = 1  # Set mask to 1 where points are visible and inside the mask
        
    # fmask = torch.zeros_like(mask)
    # fmask[py_valid, px_valid] = mask[py_valid, px_valid] 
    # import matplotlib.pyplot as plt
    # tensor_hw = fmask.cpu()  # If it's on GPU
    # plt.imshow(tensor_hw, cmap='gray')
    # plt.axis('off')
    # plt.show()
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