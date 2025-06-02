
import shutil
import dearpygui.dearpygui as dpg
import numpy as np
import random
import os, sys
import torch
from random import randint
from torchvision.utils import save_image

from tqdm import tqdm
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import itertools
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.timer import Timer

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, l1_loss_intense
from pytorch_msssim import ms_ssim
import cv2

from gaussian_renderer import render,render_batch,render_coarse_batch,render_coarse_batch_target
import json
import open3d as o3d
# from submodules.DAV2.depth_anything_v2.dpt import DepthAnythingV2
from gui_utils.base import get_in_view_dyn_mask


def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))
def minmax_normalize_nonzero(patches):
    # Set zeros to very high/low values so they don't affect min/max
    patches = patches.clone()
    patches_new = torch.zeros_like(patches, device=patches.device)
    for index, patch in enumerate(patches):
        if patch.mean() > 0.:
            patch_mask = patch > 0
            min_val = patch.min()
            max_val = patch.max()
            
            # min_val = patch[patch_mask].min()
            # max_val = patch[patch_mask].max()
            
            norm_patch = (patch - min_val) / (max_val - min_val)
            # patches_new[index, patch_mask] = norm_patch[patch_mask]
            patches_new[index, patch_mask] = norm_patch[patch_mask]

    return patches_new
def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def unpatchify(patches, image_size, patch_size):
    # patches: (num_patches, patch_area)
    B = 1
    C = 1
    H, W = image_size
    num_patches = patches.size(0)
    patches = patches.view(B, -1, patch_size * patch_size).permute(0, 2, 1)
    output = F.fold(patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    # Correct for patch overlap (we assume no overlap here)
    divisor = F.fold(torch.ones_like(patches), output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    return (output / divisor).squeeze()

# def margin_l2_loss(network_output, gt, margin, return_mask=False):
#     mask = (network_output - gt).abs() > margin
#     if not return_mask:
#         return ((network_output - gt)[mask] ** 2).mean()
#     else:
#         return ((network_output - gt)[mask] ** 2).mean(), mask
    
# def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
#     input_patches = normalize(patchify(input, patch_size))
#     target_patches = normalize(patchify(target, patch_size))
#     return margin_l2_loss(input_patches, target_patches, margin, return_mask)

# def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
#     input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
#     target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
#     return margin_l2_loss(input_patches, target_patches, margin, return_mask)

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

from gui_utils.base import GUIBase
class GUI(GUIBase):
    def __init__(self, 
                 args, 
                 hyperparams, 
                 dataset, 
                 opt, 
                 pipe, 
                 testing_iterations, 
                 saving_iterations,
                 ckpt_start,
                 debug_from,
                 expname,
                 skip_coarse,
                 view_test
                 ):

        if skip_coarse is not None:
            self.skip_coarse = os.path.join('skip_coarse',skip_coarse)
            if os.path.exists(self.skip_coarse):
                self.stage = 'fine'
                dataset.sh_degree = 3

        else:
            self.skip_coarse = None
            self.stage = 'coarse'

        expname = 'output/'+expname
        self.expname = expname
        self.opt = opt
        self.pipe = pipe
        self.dataset = dataset
        self.dataset.model_path = expname
        self.hyperparams = hyperparams
        self.args = args
        self.args.model_path = expname
        use_gui = True
        self.saving_iterations = saving_iterations
        self.checkpoint = ckpt_start
        self.debug_from = debug_from

        self.total_frames = 300
        
        self.results_dir = os.path.join(self.args.model_path, 'active_results')
        if ckpt_start is None:
            if not os.path.exists(self.args.model_path):os.makedirs(self.args.model_path)   

            if os.path.exists(self.results_dir):
                print(f'[Removing old results] : {self.results_dir}')
                shutil.rmtree(self.results_dir)
            os.mkdir(self.results_dir)    
            
        # Set the background color
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Set the gaussian mdel and scene
        gaussians = GaussianModel(dataset.sh_degree, hyperparams)
        if ckpt_start is not None:
            scene = Scene(dataset, gaussians, args.cam_config, load_iteration=ckpt_start)
            self.stage = 'fine'
        else:
            if skip_coarse:
                gaussians.active_sh_degree = dataset.sh_degree
            scene = Scene(dataset, gaussians, args.cam_config, skip_coarse=self.skip_coarse)
        
        self.total_frames = scene.maxframes
        # Initialize DPG      
        super().__init__(use_gui, scene, gaussians, self.expname, view_test)

        # Initialize training
        self.timer = Timer()
        self.timer.start()
        self.init_taining()
        
        if skip_coarse:
            self.iteration = 1
        if ckpt_start: self.iteration = int(self.scene.loaded_iter) + 1
    
        # Initialize RGB to Depth model (DepthAnything v2)
        # model_configs = {
        #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        #     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        # }

        # encoder = 'vitb' # or 'vits', 'vitb', 'vitg'
        # self.depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 1.})
        # self.depth_model.load_state_dict(torch.load(f'submodules/DAV2/checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth', map_location='cpu'))
        # self.depth_model = self.depth_model.cuda().eval()            

    def init_taining(self):

        if self.stage == 'fine':
            self.scene.init_fine()
            self.final_iter = self.opt.iterations
        else:
            self.final_iter = self.opt.coarse_iterations

        first_iter = 1

        # Set up gaussian training
        self.gaussians.training_setup(self.opt)
        # Load from fine model if it exists

        if self.checkpoint:
            if self.stage == 'fine':
                (model_params, first_iter) = torch.load(f'{self.expname}/chkpnt_fine_{self.checkpoint}.pth')
                self.gaussians.restore(model_params, self.opt)
                
        if self.skip_coarse:
            print('Restoring coarse opt params')
            (model_params, first_iter) = torch.load(os.path.join(self.skip_coarse,'checkpoint.pth'))
            self.gaussians.restore(model_params, self.opt)
            self.iteration = 0

        # Set current iteration
        self.iteration = first_iter

        # Events for counting duration of step
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)

        if self.view_test == False:
            self.test_viewpoint_stack = self.scene.getTestCameras()
            self.random_loader  = True

            if self.stage == 'fine':
                print('Loading Fine (t = any) dataset')
                # self.scene.getTrainCameras().dataset.get_mask = True

                self.viewpoint_stack = self.scene.getTrainCameras()
                self.loader = iter(DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                    num_workers=8, collate_fn=list))
                
                viewpoint_stack = [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs] * 100
                self.filter_3D_stack = viewpoint_stack.copy()
                # self.scene.getTrainCameras().dataset.get_mask = False
            if self.stage == 'coarse': 
                print('Loading Coarse (t=0) dataset')
                self.scene.getTrainCameras().dataset.get_mask = True
                self.viewpoint_stack = [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs] * 100
                self.coarse_viewpoint_stack = [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs] * 100

                self.scene.getTrainCameras().dataset.get_mask = False
                self.loader = iter(DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                    num_workers=8, collate_fn=list))
        
                self.coarse_loader = iter(DataLoader(self.coarse_viewpoint_stack, batch_size=1, shuffle=self.random_loader,
                                                    num_workers=8, collate_fn=list))
                self.filter_3D_stack = self.viewpoint_stack.copy()
                self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)

                
    @property
    def get_zero_cams(self):
        self.scene.getTrainCameras().dataset.get_mask = True
        zero_cams = [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]
        self.scene.getTrainCameras().dataset.get_mask = False
        return zero_cams
    
    @property
    def get_batch_views(self, stack=None):
        try:
            viewpoint_cams = next(self.loader)
        except StopIteration:
            viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                num_workers=8, collate_fn=list)
            self.loader = iter(viewpoint_stack_loader)
            viewpoint_cams = next(self.loader)
        
        return viewpoint_cams
    
    def train_background_step(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.iter_start.record()
        if self.iteration < self.opt.iterations:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            
        # Update Gaussian lr for current iteration
        self.gaussians.update_learning_rate(self.iteration)          
        
        # Sample from the static cameras for background
        try:
            viewpoint_cams = next(self.coarse_loader)
        except StopIteration:
            viewpoint_stack_loader = DataLoader(self.coarse_viewpoint_stack, batch_size=1, shuffle=self.random_loader,
                                                num_workers=8, collate_fn=list)
            self.coarse_loader = iter(viewpoint_stack_loader)
            viewpoint_cams = next(self.coarse_loader)

        
        L1 = torch.tensor(0.).cuda()
        L1 = render_coarse_batch(
            viewpoint_cams, 
            self.gaussians, 
            self.pipe,
            self.background, 
            stage=self.stage,
            iteration=self.iteration
        )
        
        
        hopacloss = 0.01*((1.0 - self.gaussians.get_hopac)**2).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()
        wopacloss = ((self.gaussians.get_wopac).abs()).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()

        loss = L1 + hopacloss + wopacloss

        # print( planeloss ,depthloss,hopacloss ,wopacloss ,normloss ,pg_loss,covloss)
        with torch.no_grad():
            if self.gui:
                    dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                    dpg.set_value("_log_opacs", f"h/w: {hopacloss}  |  {wopacloss} ")
                    if (self.iteration % 2) == 0:
                        dpg.set_value("_log_points", f"Scene Pts: {(~self.gaussians.target_mask).sum()} | Target Pts: {(self.gaussians.target_mask).sum()} ")
                    
            if self.iteration % 2000 == 0:
                self.track_cpu_gpu_usage(0.1)
            # Error if loss becomes nan
            if torch.isnan(loss).any():
                print("loss is nan, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
        # Backpass
        loss.backward()
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        self.iter_end.record()
        
        with torch.no_grad():
            self.timer.pause()
            torch.cuda.synchronize()
            
            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations) or (self.iteration == self.final_iter-1):
                self.save_scene()
            self.timer.start()

            if self.iteration % 100 == 0 and self.iteration < self.final_iter - 200:
                self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)

            # if self.iteration % self.opt.opacity_reset_interval == 0:
            #     self.gaussians.reset_opacity_background()
            
    def train_foreground_step(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.iter_start.record()
        if self.iteration < self.opt.iterations:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            
        # Update Gaussian lr for current iteration
        self.gaussians.update_learning_rate(self.iteration)          
        
        viewpoint_cams = self.get_batch_views

        L1 = torch.tensor(0.).cuda()
        L1 = render_coarse_batch_target(
            viewpoint_cams, 
            self.gaussians, 
            self.pipe,
            self.background, 
            stage=self.stage,
            iteration=self.iteration
        )
        
        loss = L1

        # print( planeloss ,depthloss,hopacloss ,wopacloss ,normloss ,pg_loss,covloss)
        with torch.no_grad():
            if self.gui:
                    dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                    if (self.iteration % 2) == 0:
                        dpg.set_value("_log_points", f"Scene Pts: {(~self.gaussians.target_mask).sum()} | Target Pts: {(self.gaussians.target_mask).sum()} ")
                    
            if self.iteration % 2000 == 0:
                self.track_cpu_gpu_usage(0.1)
            # Error if loss becomes nan
            if torch.isnan(loss).any():
                print("loss is nan, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
        # Backpass
        loss.backward()
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        self.iter_end.record()
        
        with torch.no_grad():
            self.timer.pause()
            torch.cuda.synchronize()
            
            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations) or (self.iteration == self.final_iter-1):
                self.save_scene()
            self.timer.start()

            if self.iteration % 100 == 0 and self.iteration < self.final_iter - 200:
                self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)

            # if self.iteration % self.opt.opacity_reset_interval == 0:
            #     self.gaussians.reset_opacity_background()
            
    def train_step(self):

        # Start recording step duration
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.iter_start.record()
        
        if self.iteration < self.opt.iterations:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        # Update Gaussian lr for current iteration
        self.gaussians.update_learning_rate(self.iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 100 == 0:
            self.gaussians.oneupSHdegree()
        
        # if self.stage == 'fine' and self.iteration % 1000 == 0 and self.iteration > 1:
        #     self.gaussians.update_wavelevel()
           
        viewpoint_cams = self.get_batch_views
        # print(self.iteration)
        if self.iteration == 1 : #or self.iteration == 4000:
            self.gaussians.dupelicate()
            self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)

        # Generate scene based on an input camera from our current batch (defined by viewpoint_cams)
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        L1 = torch.tensor(0.).cuda()
        
        radii_list, visibility_filter_list, viewspace_point_tensor_list, L1, extra_losses = render_batch(
            viewpoint_cams, 
            self.gaussians, 
            self.pipe,
            self.background, 
            stage=self.stage,
            iteration=self.iteration
        )

        depthloss, normloss, covloss, target_depth = extra_losses
        
        # hopacloss = 0.01*((1.0 - self.gaussians.get_hopac)**2).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()
        # wopacloss = ((self.gaussians.get_wopac).abs()).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()

        scale_exp = self.gaussians.get_scaling_with_3D_filter
        # pg_loss = 0.001*(scale_exp.max(dim=1).values / scale_exp.min(dim=1).values).mean()
        max_gauss_ratio = 10
        # scale_exp = self.gaussians.get_scaling
        pg_loss = (
            torch.maximum(
                scale_exp.amax(dim=-1)  / scale_exp.amin(dim=-1),
                torch.tensor(max_gauss_ratio),
            )
            - max_gauss_ratio
        ).mean()
        # pg_loss = 0.

        if self.stage == 'coarse':
            planeloss = 0.

            # normloss += self.gaussians.compute_static_rigidity_loss(self.iteration)
            
            
            
        elif self.stage == 'fine':
            # dyn_target_loss += self.gaussians.compute_rigidity_loss(self.iteration)
            
            planeloss = self.gaussians.compute_regulation(
                self.hyperparams.time_smoothness_weight, self.hyperparams.l1_time_planes, self.hyperparams.plane_tv_weight,
                self.hyperparams.minview_weight, self.hyperparams.tvtotal1_weight, 
                self.hyperparams.spsmoothness_weight, self.hyperparams.minmotion_weight
            )
            # max_gauss_ratio = 10
            # pg_loss = (
            #     torch.maximum(
            #         scale_exp.amax(dim=-1)  / scale_exp.amin(dim=-1),
            #         torch.tensor(max_gauss_ratio),
            #     )
            #     - max_gauss_ratio
            # ).mean()
            # depth_loss = render_batch_displacement(
            #     self.get_batch_views, 
            #     self.gaussians, 
            #     self.pipe,
            #     self.background,
            # )

        loss = L1 +  planeloss + \
            depthloss +\
                normloss + \
                       pg_loss + \
                           covloss
                # hopacloss + wopacloss +\
                   
        # print( planeloss ,depthloss,hopacloss ,wopacloss ,normloss ,pg_loss,covloss)
        with torch.no_grad():
            if self.gui:
                    dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                    dpg.set_value("_log_loss", f"Loss: {L1.item()} | Planes {planeloss} ")
                    # dpg.set_value("_log_opacs", f"h/w: {hopacloss}  |  {wopacloss} ")
                    dpg.set_value("_log_depth", f"PhysG: {pg_loss} ")
                    dpg.set_value("_log_dynscales", f"Norms : {normloss} ")
                    dpg.set_value("_log_knn", f"Depth: {depthloss} | cov {covloss} ")
                    if (self.iteration % 2) == 0:
                        dpg.set_value("_log_points", f"Scene Pts: {(~self.gaussians.target_mask).sum()} | Target Pts: {(self.gaussians.target_mask).sum()} ")
                    

            if self.iteration % 2000 == 0:
                self.track_cpu_gpu_usage(0.1)
            # Error if loss becomes nan
            if torch.isnan(loss).any():
                    
                print("loss is nan, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
        # Backpass
        loss.backward()
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        self.iter_end.record()
        
        with torch.no_grad():
            # radii = torch.cat(radii_list, 0).max(dim=0).values
            # visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            
            # viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[0])
            # for idx in range(0, len(viewspace_point_tensor_list)):
            #     viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            
            self.timer.pause() # log and save
           
            torch.cuda.synchronize()
            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations) or (self.iteration == self.final_iter-1):
                self.save_scene()

            self.timer.start()
            
            # Keep track of max radii in image-space for pruning
            # self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # self.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

            # Densification for scene
            # densify_threshold = self.opt.densify_grad_threshold
            # if  self.stage == 'fine' and \
            #     self.iteration > self.opt.densify_from_iter and \
            #     self.iteration < self.opt.densify_until_iter and \
            #     self.iteration % self.opt.densification_interval == 0 and \
            #     (self.gaussians.target_mask).sum() < 100000:
            #         self.gaussians.densify(densify_threshold,self.scene.cameras_extent)
            #         self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)
            #         # self.gaussians.split_spikes()

            # # Global pruning
            # if  self.stage == 'fine' and \
            #     self.iteration > self.opt.pruning_from_iter and \
            #     self.iteration % self.opt.pruning_interval == 0 and \
            #     (self.gaussians.target_mask).sum() > 100000:
            #     size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
            #     self.gaussians.prune(self.hyperparams.opacity_lambda, self.scene.cameras_extent, size_threshold)
            #     self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)
            
            if self.iteration % 100 == 0 and self.iteration < self.final_iter - 200:
                self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)
            
            # if self.stage == 'coarse' and self.iteration == 300:
            #     self.gaussians.densify_coarse(viewpoint_cams[-1],target_depth)
            # if self.stage == 'coarse' and self.iteration % 1000:
            #     self.gaussians.prune_coarse()
            # if self.iteration == 1 and self.stage == 'fine':
            #     self.gaussians.prune_target(self.get_zero_cams)

            if self.iteration % self.opt.opacity_reset_interval == 0 and self.stage == 'fine':
                self.gaussians.reset_opacity()
                
            # Optimizer step
            # if self.iteration  == 3000:
            #     self.gaussians.densify_target(zero_cams)

            
            # if self.iteration < self.opt.iterations:
            #     self.gaussians.optimizer.step()
            #     self.gaussians.optimizer.zero_grad(set_to_none = True)

            # if self.stage == 'coarse' and self.iteration == 2000:
            #     cam_list= self.get_zero_cams
            #     self.gaussians.update_target_mask(cam_list)
            #     self.final_iter += 1000

    @torch.no_grad()
    def test_step(self):
        print('testing')
        if self.iteration < (self.final_iter -1) and (self.iteration % 500) != 0 and self.view_test == False:
            idx = 0
            viewpoint_cams = []
            
            # Construct batch for current step
            while idx < 10:
                # If viewpoint stack doesn't exist get it from the temp view point list
                if not self.test_viewpoint_stack:
                    self.test_viewpoint_stack = self.scene.getTestCameras().copy()

                viewpoint_cams.append(self.test_viewpoint_stack[randint(0,len(self.test_viewpoint_stack)-1)]) # TODO: perhaps ensuring varying view positions rather thn just random views
                idx +=1
        else:
            self.test_viewpoint_stack = self.scene.getTestCameras() #.copy()
            viewpoint_cams = self.test_viewpoint_stack


        # if not os.path.exists(f'debugging/{self.iteration}') and self.iteration % 1000 == 0:
        #     os.mkdir(f'debugging/{self.iteration}')

        PSNR = 0.
        SSIM = 0.
        fullPSNR = 0.
        fullSSIM = 0.
        dumbPSNR = 0.

        idx = 0
        # Render and return preds
        for viewpoint_cam in tqdm(viewpoint_cams, desc="Processing viewpoints"):
            try: # If we have seperate depth
                viewpoint_cam, depth_cam = viewpoint_cam
            except:
                depth_cam = None

            render_pkg = render(
                viewpoint_cam, 
                self.gaussians, 
                self.pipe, 
                self.background, 
                stage='test-full'
            )
            image = render_pkg["render"]


            gt_image = viewpoint_cam.original_image.cuda()
            mask = viewpoint_cam.gt_alpha_mask.cuda()

            if (self.iteration == 500) or (self.iteration == 1000 or self.iteration == 2000):
                if idx % 100 == 0:
                    save_gt_pred(gt_image, image, self.iteration, idx, self.args.expname.split('/')[-1])
            
            if self.view_test or self.iteration > (self.final_iter -2):
                save_gt_pred_full(gt_image, image, self.iteration, idx, self.args.expname)

            fullPSNR += psnr(image, gt_image)
            fullSSIM += ssim(image.unsqueeze(0), gt_image)
            
            render_pkg = render(
                viewpoint_cam, 
                self.gaussians, 
                self.pipe, 
                self.background, 
                stage='test-foreground'
            )
            image = render_pkg["render"]
            gt_image = gt_image*mask
            image = image*mask
            PSNR += psnr(image, gt_image)
            SSIM += ssim(image.unsqueeze(0), gt_image)
            
            idx += 1

            if idx % 4 == 0 and self.gui:
                with torch.no_grad():
                    self.viewer_step()
                    dpg.render_dearpygui_frame()


        # Loss
        # dumbPSNR = dumbPSNR.item()/len(viewpoint_cams)
        fullPSNR = fullPSNR.item()/len(viewpoint_cams)
        fullSSIM = fullSSIM/len(viewpoint_cams)
        PSNR = PSNR.item()/len(viewpoint_cams)
        SSIM = SSIM/len(viewpoint_cams)

        save_file = os.path.join(self.results_dir, f'{self.iteration}.json')
        with open(save_file, 'w') as f:
            obj = {
                'full-psnr': fullPSNR,
                'full-ssim': fullSSIM.item(),
                'psnr': PSNR,
                'ssim': SSIM.item(),
                'points':self.gaussians._xyz.shape[0]}
            json.dump(obj, f)


        # Only compute extra metrics at the end of training -> can be slow
        if self.gui:
            if self.iteration < (self.final_iter -1):
                dpg.set_value("_log_psnr_test", "PSNR : {:>12.7f}".format(PSNR, ".5"))
                dpg.set_value("_log_ssim", "SSIM : {:>12.7f}".format(SSIM, ".5"))

            else:
                dpg.set_value("_log_psnr_test", "PSNR : {:>12.7f}".format(PSNR, ".5"))
                dpg.set_value("_log_ssim", "SSIM : {:>12.7f}".format(SSIM, ".5"))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def save_gt_pred_full(gt, pred, iteration, idx, name):

    pred = (pred.permute(1, 2, 0)
        # .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )*255

    gt = (gt.permute(1, 2, 0)
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
    )*255

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)

    # image_array = np.hstack((gt_bgr, pred_bgr))
    if not os.path.exists(f'output/{name}/images/'):
        os.mkdir(f'output/{name}/images/')
    cv2.imwrite(f'output/{name}/images/{idx}.png', pred_bgr)

    return pred_bgr

def save_gt_pred(gt, pred, iteration, idx, name):
    pred = (pred.permute(1, 2, 0)
        # .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )*255

    gt = (gt.permute(1, 2, 0)
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
    )*255

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)

    # image_array = np.hstack((gt_bgr, pred_bgr))

    print(f'debugging/{iteration}_{name}_{idx}.png')
    cv2.imwrite(f'debugging/{iteration}_{name}_{idx}.png', pred_bgr)

    return pred_bgr

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[8000, 9999,15999, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument('--skip-coarse', type=str, default = None)
    parser.add_argument('--view-test', action='store_true', default=False)
    parser.add_argument("--cam-config", type=str, default = "4")
    
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
        
    
    torch.autograd.set_detect_anomaly(True)
    hyp = hp.extract(args)
    initial_name = args.expname     
    name = f'{initial_name}'
    gui = GUI(
        args=args, 
        hyperparams=hyp, 
        dataset=lp.extract(args), 
        opt=op.extract(args), 
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations, 
        saving_iterations=args.save_iterations,
        ckpt_start=args.start_checkpoint, 
        debug_from=args.debug_from, 
        expname=name,
        skip_coarse=args.skip_coarse,
        view_test=args.view_test
    )
    gui.render()
    del gui
    torch.cuda.empty_cache()
    # TV Reg
    # hyp.plane_tv_weight = 0.
    # for value in [0.001,0.00075,0.00025,0.0001,]:
    #     name = f'{initial_name}_TV{value}'
    #     hyp.plane_tv_weight = value
        
    #     # Start GUI server, configure and run training
    #     gui = GUI(
    #         args=args, 
    #         hyperparams=hyp, 
    #         dataset=lp.extract(args), 
    #         opt=op.extract(args), 
    #         pipe=pp.extract(args),
    #         testing_iterations=args.test_iterations, 
    #         saving_iterations=args.save_iterations,
    #         ckpt_start=args.start_checkpoint, 
    #         debug_from=args.debug_from, 
    #         expname=name,
    #         skip_coarse=args.skip_coarse,
    #         view_test=args.view_test
    #     )

        
    #     gui.render()
    #     del gui
    #     torch.cuda.empty_cache()
    #     print("\nTraining complete.")
    
    # # Spatial smoothness
    # hyp.tvtotal1_weight = 0.
    # for value in [0.1,0.01,0.001,0.0001,0.00001]:
    #     name = f'{initial_name}_SP{value}'
    #     hyp.spsmoothness_weight = value
        
    #     gui = GUI(
    #         args=args, 
    #         hyperparams=hyp, 
    #         dataset=lp.extract(args), 
    #         opt=op.extract(args), 
    #         pipe=pp.extract(args),
    #         testing_iterations=args.test_iterations, 
    #         saving_iterations=args.save_iterations,
    #         ckpt_start=args.start_checkpoint, 
    #         debug_from=args.debug_from, 
    #         expname=name,
    #         skip_coarse=args.skip_coarse,
    #         view_test=args.view_test
    #     )
    #     gui.render()
    #     del gui
    #     torch.cuda.empty_cache()
    #     print("\nTraining complete.")
    
    # Minview Weight
    # hyp.spsmoothness_weight = 0.
    # hyp.plane_tv_weight = 0.0005
    
    # for value in [0.01,0.001,0.0001,0.00001]:
    #     name = f'{initial_name}_Angle{value}'
    #     hyp.minview_weight = value
        
    #     gui = GUI(
    #         args=args, 
    #         hyperparams=hyp, 
    #         dataset=lp.extract(args), 
    #         opt=op.extract(args), 
    #         pipe=pp.extract(args),
    #         testing_iterations=args.test_iterations, 
    #         saving_iterations=args.save_iterations,
    #         ckpt_start=args.start_checkpoint, 
    #         debug_from=args.debug_from, 
    #         expname=name,
    #         skip_coarse=args.skip_coarse,
    #         view_test=args.view_test
    #     )
    #     gui.render()
    #     del gui
    #     torch.cuda.empty_cache()
    #     print("\nTraining complete.")
    
    # Mintemporal Weight
    # hyp.minview_weight = 0.
    # hyp.l1_time_planes = 0.
    # hyp.time_smoothness_weight = 0.
    
    # for value in [0.0001,0.00001]:
    #     name = f'{initial_name}_Temporal{value}'
    #     hyp.minmotion_weight = value
        
    #     gui = GUI(
    #         args=args, 
    #         hyperparams=hyp, 
    #         dataset=lp.extract(args), 
    #         opt=op.extract(args), 
    #         pipe=pp.extract(args),
    #         testing_iterations=args.test_iterations, 
    #         saving_iterations=args.save_iterations,
    #         ckpt_start=args.start_checkpoint, 
    #         debug_from=args.debug_from, 
    #         expname=name,
    #         skip_coarse=args.skip_coarse,
    #         view_test=args.view_test
    #     )
    #     gui.render()
    #     del gui
    #     torch.cuda.empty_cache()
    #     print("\nTraining complete.")