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
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness,compute_plane_tv

from gaussian_renderer import render_motion_point_mask

from scene.gaussians.B_Gaussians import BackgroundGaussians
from scene.gaussians.F_Gaussians import ForegroundGaussians

from scipy.spatial import KDTree
import torch

def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

class GaussianScene:
    
    def __init__(self, sh_degree : int, args):
        self.fg = ForegroundGaussians(sh_degree, args)
        self.bg = BackgroundGaussians(sh_degree, args)
    
    def step_optimizers(self):
        self.fg.optimizer.step()
        self.bg.optimizer.step()
        self.fg.optimizer.zero_grad(set_to_none = True)
        self.bg.optimizer.zero_grad(set_to_none = True)

    def save_checkpoint(self, iteration, stage, path):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        self.fg.save_gaussians(iteration, stage, path)
        self.bg.save_gaussians(iteration, stage, path)
    
    def load_gaussians(self, path, stage, opt, iteration):
        self.fg.load_gaussians(path, stage, opt, iteration)
        self.bg.load_gaussians(path, stage, opt, iteration)
    
    def training_setup(self, opt):
        self.fg.training_setup(opt)
        self.bg.training_setup(opt)
        
    def create_from_pcd(self, pcd : BasicPointCloud, cam_list=None, dataset_type="dynerf"):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
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
        self.fg._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
        
        xyz_min = fused_point_cloud[~target_mask].min(0).values
        xyz_max = fused_point_cloud[~target_mask].max(0).values
        self.bg._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
        
        features = torch.zeros((fused_color.shape[0], 3, (self.fg.max_sh_degree + 1) ** 2)).float().cuda()
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
        
        self.fg.create_from_pcd(
            fused_point_cloud[target_mask], 
            features[target_mask],
            scales[target_mask],
            rots[target_mask],
            opacities[target_mask],
        )
        
        if dataset_type == 'dynerf':
            lrscaller = 0.1
        else:
            lrscaller = 1.
        self.bg.create_from_pcd(
            fused_point_cloud[~target_mask], 
            features[~target_mask],
            scales[~target_mask],
            rots[~target_mask],
            opacities[~target_mask],
            lr_scaller = lrscaller
        )
        
        print(f"Target lr scale: {self.fg.spatial_lr_scale} | Background lr scale: {self.bg.spatial_lr_scale}")

    def load_deformation(self, path):
        self.fg.load_deformation(path)
        self.bg.load_deformation(path)
    
    def save_deformation(self, path):
        self.fg.save_deformation(path)
        self.bg.save_deformation(path)
         
    def load_ply(self, path):
        self.fg.load_ply(path+'_foreground.ply')
        self.bg.load_ply(path+'_background.ply')
        
    def save_ply(self, path):
        self.fg.save_ply(path+'_foreground.ply')
        self.bg.save_ply(path+'_background.ply')

    def compute_3D_filter(self,cameras=None):
        self.fg.compute_3D_filter(cameras)
        self.bg.compute_3D_filter(cameras)
    
    def update_learning_rate(self,iteration):
        self.fg.update_learning_rate(iteration)
        self.bg.update_learning_rate(iteration)
    
    def duplicate(self, type_='default'):
        if type_ == 'default':
            self.fg.dupelicate()
        elif type_ == 'dynamic':
            self.fg.dynamic_dupelication()
        else:
            raise f'No implementation of duplication type {type_}'

    def reset_opacity(self):
        self.fg.reset_opacity()
        self.bg.reset_opacity()
        
    def get_temporal_parameters(self, viewpoint_camera, datasettype):
        
        means3D_fg, scales_fg, rotations_fg, colors_fg, opacity_fg = self.fg.get_temporal_parameters(viewpoint_camera.time)
        means3D_bg, scales_bg, rotations_bg, colors_bg, opacity_bg = self.bg.get_temporal_parameters(viewpoint_camera.time)
        
        if datasettype == 'condense':
            distances = torch.norm(means3D_bg - viewpoint_camera.camera_center.cuda(), dim=1)
            mask = distances > 0.3
            means3D_bg = means3D_bg[mask]
            rotations_bg = rotations_bg[mask]
            scales_bg = scales_bg[mask]
            opacity_bg = opacity_bg[mask]
            colors_bg = colors_bg[mask]
            
        
        return torch.cat([means3D_fg, means3D_bg], dim=0), torch.cat([scales_fg, scales_bg], dim=0),\
            torch.cat([rotations_fg, rotations_bg], dim=0),torch.cat([colors_fg, colors_bg], dim=0),\
            torch.cat([opacity_fg, opacity_bg], dim=0)
    
    def get_full_canonical_parameters(self,):
        means3D_fg, scales_fg, rotations_fg, colors_fg, opacity_fg = self.fg.get_canonical_parameters()
        means3D_bg, scales_bg, rotations_bg, colors_bg, opacity_bg = self.bg.get_canonical_parameters()
        return torch.cat([means3D_fg, means3D_bg], dim=0), torch.cat([scales_fg, scales_bg], dim=0),\
            torch.cat([rotations_fg, rotations_bg], dim=0),torch.cat([colors_fg, colors_bg], dim=0),\
            torch.cat([opacity_fg, opacity_bg], dim=0)
    
    def compute_regulation(self, time_smoothness_weight, plane_tv_weight,
                           l1_time_planes_weight,l1_col_planes_weight, 
                           tv_background_weight, ts_background_weight):
        tvtotal = 0
        l1total = 0
        tstotal = 0
        col=0
        
        tvbackground = 0
        tsbackground = 0
        
        wavelets = self.fg._deformation.deformation_net.grid.waveplanes_list()
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for index, grids in enumerate(self.fg._deformation.deformation_net.grid.grids_()):
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
                    
        for index, grids in enumerate(self.bg._deformation.deformation_net.grid.grids_()):
            if index in [0,1,3]: # space only
                for grid in grids:
                    tvbackground += compute_plane_smoothness(grid)
            elif index in [2, 4, 5]:
                for grid in grids: # space time
                    tsbackground += compute_plane_smoothness(grid)       
        
        return plane_tv_weight * tvtotal + time_smoothness_weight*tstotal + l1_time_planes_weight*l1total + l1_col_planes_weight*col +\
            ts_background_weight*tsbackground + tv_background_weight*tvbackground

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
        
    # fmask = torch.zeros_like(mask)
    # fmask[py_valid, px_valid] = mask[py_valid, px_valid] 
    # import matplotlib.pyplot as plt
    # tensor_hw = fmask.cpu()  # If it's on GPU
    # plt.imshow(tensor_hw, cmap='gray')
    # plt.axis('off')
    # plt.show()
    return final_mask 

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