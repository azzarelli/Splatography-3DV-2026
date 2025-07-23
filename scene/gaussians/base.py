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

from gaussian_renderer import render_motion_point_mask

from torch_cluster import knn_graph

class GaussianBase:
    
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
        
    def __init__(self, deformation, sh_degree, args, no_dopac=False, no_shs=False, name='default'):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        
        self._deformation = deformation 

        self._xyz = torch.empty(0)
        
        self._opacityh = torch.empty(0)
        self._opacityw = torch.empty(0)
        self._opacitymu = torch.empty(0)
        
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        
        self.filter_3D = torch.empty(0)
        
        self.spatial_lr_scale = 0.0
        
        self.no_dopac = no_dopac # TODO: implement no dynamic color for bckgrnd
        self.no_shs = no_shs # TODO: no view-dependant color for background.
        
        self.optimizer = None

        self.name = name
        self.setup_functions()

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

    def get_canonical_parameters(self, mask='none', cam_center=None, distance=0.):
        if mask == 'none':
            return self._xyz, self.get_scaling_with_3D_filter, self.get_rotation, self.get_features, self.get_fine_opacity_with_3D_filter(self.get_hopac)

        elif mask == 'distance':
            means3D, scales, rotations, colors, opacity = self._xyz, self.get_scaling_with_3D_filter, self.get_rotation, self.get_features, self.get_fine_opacity_with_3D_filter(self.get_hopac)
            
            distances = torch.norm(means3D - cam_center, dim=1)
            mask = distances > 0.3
            
            means3D = means3D[mask]
            rotations = rotations[mask]
            scales = scales[mask]
            colors = colors[mask]
            opacity = opacity[mask]
            
            return means3D, scales, rotations, colors, opacity
    
    def get_temporal_parameters(self,time):
        scales = self.get_scaling_with_3D_filter
        
        means3D, rotations, _, colors  = self._deformation(
            point=self._xyz, 
            rotations=self._rotation,
            scales=scales, 
            shs=self.get_features, 
            h_emb=self.get_opacity,
            time=time
        )

        opacity = self.get_fine_opacity_with_3D_filter(self.get_hopac)
        rotations = self.rotation_activation(rotations)
        
        return means3D, scales, rotations, colors, opacity
        
    
    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        #TODO consider focal length and image width
        xyz = self._xyz
        
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
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
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
            self.no_dopac,
            self.no_shs,
            self.name
        )
    
    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
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
            self.spatial_lr_scale,
            self.no_dopac,
            self.no_shs,
            self.name
        ) = model_args

        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def create_from_pcd(self, xyz, cols, scales, rots, opacs, lr_scaller=1.):
        self.active_sh_degree = 0
        self._deformation = self._deformation.to("cuda")

        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        self._features_dc = nn.Parameter(cols[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(cols[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacityh  = nn.Parameter(opacs[:, 0].unsqueeze(-1).requires_grad_(True))
        
        if self.no_dopac == False:
            self._opacityw  = nn.Parameter(opacs[:, 1].unsqueeze(-1).requires_grad_(True))
            self._opacitymu = nn.Parameter(opacs[:, 2].unsqueeze(-1).requires_grad_(True))

        mean_foreground = xyz.mean(dim=0).unsqueeze(0)
        dist_foreground = torch.norm(xyz - mean_foreground, dim=1)
        self.spatial_lr_scale = torch.max(dist_foreground).detach().cpu().numpy() * lr_scaller
        
    def training_setup(self, training_args):
    
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},

            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            
            {'params': [self._opacityh], 'lr': training_args.opacityh_lr, "name": "opacityh"},            
                       
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                    
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]
        
        if self.no_dopac == False:
            l.append({'params': [self._opacityw], 'lr': training_args.opacityw_lr, "name": "opacityw"})
            l.append({'params': [self._opacitymu], 'lr': training_args.opacitymu_lr, "name": "opacitymu"})

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

            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
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
        
        if self.no_dopac == False:
            for i in range(self._opacityw.shape[1]):
                l.append('opacityw_{}'.format(i))
            for i in range(self._opacitymu.shape[1]):
                l.append('opacitymu_{}'.format(i))
        
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l
    
    def save_gaussians(self, iteration, stage, path):
        torch.save((self.capture(), iteration), path + f"/chkpnt_{self.name}_{stage}_{iteration}.pth")

    def load_gaussians(self, path, stage, opt, iteration):
        (model_params, first_iter) = torch.load(f'{path}/chkpnt_{self.name}_{stage}_{iteration}.pth')
        self.restore(model_params, opt)
    
    def load_deformation(self, path):
        print("loading deformation model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,f"deformation_{self.name}.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        
    def save_deformation(self, path):
        if os.path.exists(path) == False: os.mkdir(path)
        torch.save(self._deformation.state_dict(),os.path.join(path, f"deformation_{self.name}.pth"))
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        
        opacityh = self._opacityh.detach().cpu().numpy()
        if self.no_dopac == False:
            opacityw = self._opacityw.detach().cpu().numpy()
            opacitymu = self._opacitymu.detach().cpu().numpy()
        
        normals = np.zeros_like(xyz)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        if self.no_dopac == False:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacityh,opacityw, opacitymu, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacityh, scale, rotation), axis=1)
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
        
        if self.no_dopac == False:
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
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._opacityh = nn.Parameter(torch.tensor(opacityh, dtype=torch.float, device="cuda").requires_grad_(True))
        
        if self.no_dopac == False:
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
            
            "rotation" : new_rotation,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            }
        if self.no_dopac == False:
            d["opacityw"]= new_opacitiesw
            d["opacitymu"]= new_opacitiesmu
            
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacityh = optimizable_tensors["opacityh"]
        if self.no_dopac == False:
            self._opacityw = optimizable_tensors["opacityw"]
            self._opacitymu = optimizable_tensors["opacitymu"]
        
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        
    def reset_opacity(self):
        print(f'Resetting {self.name} opacity')
        opacities_newh = self._opacityh
        # opacities_neww = self._opacityw
        opacities_newh =  torch.logit(torch.tensor(0.05)).item() + opacities_newh * 0.
        # opacities_neww = 1.5 + opacities_neww

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_newh, "opacityh")
        self._opacityh = optimizable_tensors["opacityh"]
        