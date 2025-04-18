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
from scene.regulation import compute_plane_smoothness

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

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        
        self._opacity = torch.empty(0)
        
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
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
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
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        deform_state,
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
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs  = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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
    def get_h_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_dyn_coefs(self):
        return self._deformation.get_dyn_coefs(self.get_xyz,)

    def get_G(self, time, stage):
        means3D = self.get_xyz
        time = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0], 1)
        shs = self.get_features

        scales = self._scaling
        rot = self._rotation

        opa = self.get_h_opacity

        if stage == 'fine':
            means3D, rot, opa, shs, extras = self._deformation(
                point=means3D, 
                rotations=rot,
                shs=shs,
                times_sel=time, 
                h_emb=opa,
                target_mask=self.target_mask
            )
        
        scales = self.scaling_activation(scales + 0.)
        rot = self.rotation_activation(rot)
        
        return means3D,scales,rot,shs,opa


    def opacity_integral(self, w,h,mu):
        return gaussian_integral(w,h,mu)
    
    @property
    def dynamic_point_prob(self):
        return self._deformation.deformation_net.grid.get_dynamic_probabilities(self._xyz)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = 1. * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
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
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},            
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
        l.append('opacity')
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
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
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
            
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._p = nn.Parameter(torch.tensor(ps, dtype=torch.float, device="cuda").requires_grad_(True))
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
            # if group['name'] == 'scaling':print(stored_state)
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
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
                
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        self.target_mask = self.target_mask[valid_points_mask]
        
    def densification_postfix(self, new_xyz,new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "scaling" : new_scaling,
        "opacity": new_opacities,
        "rotation" : new_rotation,
       }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

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
        
        self.densify_and_clone(extent, grads=grads, grad_threshold=max_grad)
        self.densify_and_split(extent, grads=grads_abs, grad_threshold=max_grad)
    
    def densify_and_clone(self, scene_extent, grads,grad_threshold ):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # Only clone points in the scene not the target
        selected_pts_mask = torch.logical_and(selected_pts_mask, ~self.target_mask)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask]
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacities, new_scaling, new_rotation)

    def densify_and_split(self, scene_extent, grads, grad_threshold):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        # Only select non-target points for densification
        selected_pts_mask = torch.logical_and(selected_pts_mask, ~self.target_mask)
        
        N = 2
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        # Update target mask
        new_target_mask = self.target_mask[selected_pts_mask].repeat(N)
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)            

        self.densification_postfix(new_xyz,new_features_dc, new_features_rest,new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    
    def densify_target(self,cam_list):     
        self.clone_target()
        self.split_target()
        
        self.prune_target(cam_list)

        torch.cuda.empty_cache()
    
    def prune_target(self, cam_list):
        target_mask = torch.zeros_like(self.get_xyz[:,0],dtype=torch.long, device=self.get_xyz.device)
        # Cycle through each view and upate mask to track how many views each point is within
        for cam in cam_list:             
            target_mask += get_in_view_dyn_mask(cam, self.get_xyz).long()
        # New densified points that sit ouside the target region
        mask  = target_mask < (len(cam_list)-1)
        
        # Prune points that should be in our mask but aren't
        #   Dupilation and splitting may have forced them outside the mask
        selected_pts_mask = torch.logical_and(mask, self.target_mask)
        
        # Also prune points that are in the target region but not in the target mask 
        # scene_pts = torch.logical_and(~mask, ~self.target_mask)

        # selected_pts_mask = torch.logical_or(target_points, scene_pts)
        self.prune_points(selected_pts_mask)
    
    def clone_target(self):
        selected_pts_mask = self.target_mask
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_target_mask = self.target_mask[selected_pts_mask]
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacities, new_scaling, new_rotation)
        
    def split_target(self,):
        N = 2
        selected_pts_mask = self.target_mask
    
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        new_target_mask = self.target_mask[selected_pts_mask].repeat(N)
        self.target_mask = torch.cat([self.target_mask, new_target_mask], dim=0)
        self.densification_postfix(new_xyz,new_features_dc, new_features_rest,new_opacity, new_scaling, new_rotation)
            

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
        h = self.get_h_opacity
        # Hyper params
        # width_threshold = float(1./ (2. * self._deformation.deformation_net.grid.grid_config[0]['resolution'][3]))
        # Calulcate min width
        # width = self.compute_topac_width(w, h, 0.025)
        # prune_mask = #(width < width_threshold).squeeze()
        prune_mask = (h < 0.1).squeeze() #torch.logical_or((h < 0.1).squeeze(), prune_mask)
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            if prune_mask is None:
                prune_mask = big_points_vs
            else:
                prune_mask = torch.logical_or(prune_mask, big_points_vs)
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        
    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_h_opacity, torch.ones_like(self.get_h_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def _plane_regulation(self):
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in self._deformation.deformation_net.grid.grids_():
            time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        

        return total

    def _time_regulation(self):
    
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in self._deformation.deformation_net.grid.grids_():
            time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
                
        return total
    
    def _l1_regulation(self):
        total = 0.0
        for grids in self._deformation.deformation_net.grid.grids_():
            spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
                
        return total
    
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight ):
        return plane_tv_weight * self._plane_regulation() + \
            time_smoothness_weight * self._time_regulation() + \
                l1_time_planes_weight * self._l1_regulation() 

    def update_target_mask(self, cam_list, iteration, stage):    
        dyn_mask = torch.zeros_like(self.get_xyz[:,0],dtype=torch.long, device=self.get_xyz.device)
        for cam in cam_list:             
            dyn_mask += get_in_view_dyn_mask(cam, self.get_xyz).long()
        
        self.target_mask  = dyn_mask > (len(cam_list)-1)
        
        # if self._deformation.deformation_net.target_grid.aabb == None:
        #     points = self._xyz[self.target_mask]
        #     xyz_min = torch.min(points, dim=0).values
        #     xyz_max = torch.max(points, dim=0).values
        #     print(xyz_max)
        #     self._deformation.deformation_net.target_grid.set_aabb(xyz_max, xyz_min)
    
    def compute_rigidity_loss(self):        
        edge_index = knn_graph(self._xyz, k=3, batch=None, loop=False)
        row, col = edge_index
        
        distance_mask = 0.1 > torch.norm(self._xyz[row] - self._xyz[col], dim=1)  # (E,)
        dx_coefs = self.get_dyn_coefs
        diff = ((distance_mask*(((dx_coefs[row] - dx_coefs[col])**2).mean(-1)))).mean()
        return diff
         
         
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
   
SQRT_PI = torch.sqrt(torch.tensor(torch.pi))
def gaussian_integral(w,h, mu):
    """Returns high weight (0 to 1) for solid materials
    
        Notes on optimization:
            We evaluate the function return h - h*(SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2)
            This tends to 0 when out point is solid and can be simplified into 
            (SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2)
            
            The below function is the unit area minus the area under the curve within the bounded area without (h)
            As this value tends to 1 the gaussian is static in opactiy temporal domain
 
    """
    erf_term_1 = torch.erf(w * mu)
    erf_term_2 = torch.erf(w * (mu - 1))
    EPS = 1e-8
    fact = (SQRT_PI / ((2. * w) + EPS))
    return (fact * (erf_term_1 - erf_term_2)).squeeze(-1)

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