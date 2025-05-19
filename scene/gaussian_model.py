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
            self._colors,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
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
        self._colors,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
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
    def get_aabb(self):
        return self._deformation.get_aabb
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self._opacity
    
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


    def opacity_integral(self, w,h,mu):
        print("missing opac integral in  gaussing class")
        exit()
        return gaussian_integral(w)
    
    @property
    def dynamic_point_prob(self):
        return self._deformation.deformation_net.grid.get_dynamic_probabilities(self._xyz)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int, cam_list=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        # Construct pcd, increase number of points in target region
        # increase scene points as well
        # Remove unseen points
        # Remove points betwen 1 and  target view masks
        dyn_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()
        scene_mask = torch.zeros_like(fused_point_cloud[:,0],dtype=torch.long).cuda()
        
        for cam in cam_list:             
            dyn_mask += get_in_view_dyn_mask(cam, fused_point_cloud).long()
            scene_mask += get_in_view_screenspace(cam, fused_point_cloud).long()
        
        scene_mask = scene_mask > 0
        target_mask = dyn_mask > (len(cam_list)-1)
        dyn_mask = torch.logical_or(target_mask, dyn_mask ==0)
        viable  = torch.logical_and(dyn_mask, scene_mask)
        
        fused_point_cloud = fused_point_cloud[viable]
        fused_color = fused_color[viable]
        target_mask = target_mask[viable]

        
        while target_mask.sum() < 40000:
            target_point_noise =  fused_point_cloud[target_mask] + torch.randn_like(fused_point_cloud[target_mask]).cuda() * 0.05
            fused_point_cloud = torch.cat([fused_point_cloud,target_point_noise], dim=0)
            fused_color = torch.cat([fused_color,fused_color[target_mask]], dim=0)
            target_mask = torch.cat([target_mask, target_mask[target_mask]])
        
        self.target_mask = target_mask
        
        xyz_min = fused_point_cloud[target_mask].min(0).values - .1
        xyz_max = fused_point_cloud[target_mask].max(0).values + .1

        self._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # Initialize opacities
        opacities = 1. * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        # Set h = 1 : As max_opac = sig(h) to set max opac = 1 we need h = logit(1)
        opacities[:, 0] = torch.logit(opacities[:, 0])
        # Set w = 0.01 : As w_t = sig(w)*200, we need to set w = logit(w_t/200)
        opacities[:, 1] = (opacities[:, 1]*.05)
        # Finally set mu to 0 as the start of the traniing
        opacities[:, 2] = torch.logit(opacities[:, 2]*0.)
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._colors = nn.Parameter(fused_color.requires_grad_(True))
        self._deformation = self._deformation.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
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
            # {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},            
            {'params': [self._colors], 'lr': training_args.feature_lr, "name": "color"},            
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

        for i in range(self._colors.shape[1]):
            l.append('color_{}'.format(i))
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

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, colors, opacities, scale, rotation), axis=1)
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
            
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._colors = nn.Parameter(torch.tensor(cols, dtype=torch.float, device="cuda").requires_grad_(True))
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
        mask = torch.logical_and(mask, ~self.target_mask)
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._colors = optimizable_tensors["color"]

        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
                
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        self.target_mask = self.target_mask[valid_points_mask]
        
    def densification_postfix(self, new_xyz,new_colors, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,

        "scaling" : new_scaling,
        "opacity": new_opacities,
        "rotation" : new_rotation,
        "color":new_colors, 
       }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._colors = optimizable_tensors["color"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor[update_filter, 2:], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
    
    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        
        # self.split_spikes(extent, grads=grads, grad_threshold=max_grad)
        self.densify_and_clone(extent, grads=grads_abs, grad_threshold=max_grad)
        self.densify_and_split(extent, grads=grads_abs, grad_threshold=max_grad)
    
    def densify_and_clone(self, scene_extent, grads,grad_threshold ):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # Only clone points in the scene not the target
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.target_mask)
        
        new_xyz = self._xyz[selected_pts_mask]
        # new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_colors = self._colors[selected_pts_mask]
        
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask]
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)

        self.densification_postfix(new_xyz, new_colors,new_opacities, new_scaling, new_rotation)

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
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
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
        
    def prune(self, h_thresold, extent, max_screen_size,):
        """
        
            Notes:
                So purging H seems to work when we apply sigmoid to h and mu instead of using the opacity emebdding and activator
                
                Lets test the following:
                    We need to evaluate the time points where the temporal opacity function crosses a min(h) threshold e.g. h=0.05
                    There will be two intersections with the y = h = 0.05 and these can be determined with:
                        t = m +- sqrt(-ln(0.05/h)/(w**2))
                        
                    We can then asses the absolute distance between t_neg and t_pos and remove spikes that apear smaller than our temporal
                    grid resolution (basically temporal pruning)
        """
        h = self.get_hopac
        prune_mask = (h < 0.05).squeeze(-1)

        h_mask = torch.logical_and((h < 0.5).squeeze(-1), self.target_mask)
        prune_mask = torch.logical_or(h_mask, prune_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        
    
    def reset_opacity(self):
        print('resetting opacity')
        opacities_new = self.get_opacity
        opacities_new[:,0] = 0.01
        opacities_new[:,1] = 0.05

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
        tvtotal1 = 0
        spsmoothness = 0
        minmotion = 0
        minview = 0
        for index, grids in enumerate(self._deformation.deformation_net.grid.waveplanes_list()):
            if index in [0,1,3]: # space only
                for idx, grid in enumerate(grids):
                    # Total variation
                    if idx == 0:
                        tvtotal1 += compute_plane_tv(grid)
                    
                    # Spatial smoothness prioritizing coarse features
                    elif idx == 1:
                        spsmoothness += 0.6*grid.abs().mean()
                    elif idx == 2:
                        spsmoothness += 0.3*grid.abs().mean()
                        
            elif index in [2,4,5]: # space time
                for idx, grid in enumerate(grids):
                    # Min motion
                    if idx == 0: 
                        minmotion += grid.abs().mean()
                    elif idx == 1:
                        minmotion += 0.6 * grid.abs().mean()
                    elif idx == 2:
                        minmotion += 0.3 * grid.abs().mean()
            else:
                for idx, grid in enumerate(grids):
                    if idx == 0: 
                        minview += grid.abs().mean()
                    elif idx == 1:
                        minview += 0.6 * grid.abs().mean()
                    elif idx == 2:
                        minview += 0.3 * grid.abs().mean()
        
        return plane_tv_weight * tvtotal + time_smoothness_weight*tstotal + l1_time_planes_weight*l1total + \
            tvtotal1 * tvtotal1_weight + spsmoothness * spsmoothness_weight + minmotion * minmotion_weight + minview * minview_weight

    def update_target_mask(self, cam_list): 
        selected_pts_mask = self.target_mask
        
        noise = torch.randn_like(self._xyz[selected_pts_mask]) * 0.01 
        new_xyz = self._xyz[selected_pts_mask] + noise
        new_colors = self._colors[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask]
        
        dyn_mask = torch.zeros_like(new_xyz[:, 0],dtype=torch.long, device=self.get_xyz.device)
        for cam in cam_list:             
            dyn_mask += get_in_view_dyn_mask(cam, new_xyz).long()

        target_add  = dyn_mask > (len(cam_list)-1)
        
        new_xyz = new_xyz[target_add]
        new_colors = new_colors[target_add]
        new_scaling = new_scaling[target_add]
        new_rotation = new_rotation[target_add]
        new_opacities = new_opacities[target_add]
        new_target_mask = new_target_mask[target_add]
        
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)

        self.densification_postfix(new_xyz, new_colors,new_opacities, new_scaling, new_rotation)

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

    def update_neighbours(self,points):
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

         
def get_in_view_dyn_mask(camera, xyz: torch.Tensor) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # Apply full projection (world → clip space)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    # Homogeneous divide to get NDC coordinates
    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    # Visibility check
    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2].abs() <= 1)
    visible_mask = in_ndc_bounds & in_front
    
    # Compute pixel coordinates
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long()
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long()    # Init mask values
    mask_values = torch.zeros(N, dtype=torch.bool, device=device)

    # Only sample pixels for visible points
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]

    if valid_idx.numel() > 0:
        px_valid = px[valid_idx].clamp(0, camera.image_width - 1)
        py_valid = py[valid_idx].clamp(0, camera.image_height - 1)
        mask = camera.mask.to(device)
        sampled_mask = mask[py_valid, px_valid]  # shape: [#valid]
        mask_values[valid_idx] = sampled_mask.bool()
    # import matplotlib.pyplot as plt

    # # Assuming tensor is named `tensor_wh` with shape [W, H]
    # # Convert to [H, W] for display (matplotlib expects H first)
    # mask[py_valid, px_valid] = 0.5
    # print(py_valid.shape)

    # tensor_hw = mask.cpu()  # If it's on GPU
    # plt.imshow(tensor_hw, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # exit()
    return mask_values.long()

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