
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

from gaussian_renderer import render,render_batch,render_coarse_batch,render_coarse_batch_vanilla,render_coarse_batch_target, render_depth_batch
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
                 view_test,
                 use_gui:bool=False
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
                                                    num_workers=16, collate_fn=list))
                
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
                                                    num_workers=16, collate_fn=list))
        
                self.coarse_loader = iter(DataLoader(self.coarse_viewpoint_stack, batch_size=1, shuffle=self.random_loader,
                                                    num_workers=16, collate_fn=list))
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
                                                num_workers=16, collate_fn=list)
            self.loader = iter(viewpoint_stack_loader)
            viewpoint_cams = next(self.loader)
        
        return viewpoint_cams
    
    def train_background_step(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.iter_start.record()
        if self.iteration < self.opt.iterations:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 500 == 0:
            self.gaussians.oneupSHdegree()
            
        # Update Gaussian lr for current iteration
        self.gaussians.update_learning_rate(self.iteration)          
        
        if self.iteration == int(self.final_iter/2):
            print("Dupelication")
            self.gaussians.dupelicate()
            self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)
        
        # Sample from the static cameras for background
        try:
            viewpoint_cams = next(self.coarse_loader)
        except StopIteration:
            viewpoint_stack_loader = DataLoader(self.coarse_viewpoint_stack, batch_size=1, shuffle=self.random_loader,
                                                num_workers=8, collate_fn=list)
            self.coarse_loader = iter(viewpoint_stack_loader)
            viewpoint_cams = next(self.coarse_loader)

        
        L1 = torch.tensor(0.).cuda()
        if self.scene.dataset_type == "condense" or  self.scene.dataset_type == "dynerf":
            L1 = render_coarse_batch(
                viewpoint_cams, 
                self.gaussians, 
                self.pipe,
                self.background, 
                stage=self.stage,
                iteration=self.iteration
            )
        else:
            L1 = render_coarse_batch_vanilla(
                viewpoint_cams, 
                self.gaussians, 
            )
        
        hopacloss = 0.01*((1.0 - self.gaussians.get_hopac)**2).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()
        wopacloss = ((self.gaussians.get_wopac).abs()).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()

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
        
        loss = L1 + 0.01*(hopacloss + wopacloss) + pg_loss

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

            # if self.iteration % self.opt.opacity_reset_interval == 0 and self.iteration > 1000:
            #     self.gaussians.reset_opacity_background()
            
    def train_foreground_step(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.iter_start.record()
        if self.iteration < self.opt.iterations:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
      
        
        viewpoint_cams = self.get_batch_views
        
        loss = render_coarse_batch_target(
            viewpoint_cams, 
            self.gaussians, 
            self.pipe,
            self.background, 
            stage=self.stage,
            iteration=self.iteration
        )
        

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

    def train_foreground_fine_step(self):
        """Train canon foreground GS using the t=0 (as was initialized)
        """
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.iter_start.record()
        if self.iteration < self.opt.iterations:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        
        try:
            viewpoint_cams = next(self.coarse_loader)
        except StopIteration:
            viewpoint_stack_loader = DataLoader(self.coarse_viewpoint_stack, batch_size=1, shuffle=self.random_loader,
                                                num_workers=8, collate_fn=list)
            self.coarse_loader = iter(viewpoint_stack_loader)
            viewpoint_cams = next(self.coarse_loader)
            
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

        with torch.no_grad():
            if torch.isnan(loss).any():
                print("loss is nan, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
        loss.backward()
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        self.iter_end.record()
            
    def train_step(self):

        # Start recording step duration
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.iter_start.record()
        
        if self.iteration < self.opt.iterations:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        # Update Gaussian lr for current iteration
        self.gaussians.update_learning_rate(self.iteration)
           
        viewpoint_cams = self.get_batch_views
        
        if (self.scene.dataset_type == "dynerf" and self.iteration in [3000]) or (self.scene.dataset_type == "condense" and self.iteration in [3000, 6000]):
            print("Dupelicating Dynamics")
            self.gaussians.dynamic_dupelication()
            self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)
        
        # if self.iteration == -1 and self.scene.dataset_type == "condense": # TODO: Maybe this is unecessary?
        #     print("Dupelicating Dynamics")
        #     self.gaussians.dupelicate()
        #     self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)
        
        L1 = torch.tensor(0.).cuda()
        
        L1 = render_batch(
            viewpoint_cams, 
            self.gaussians, 
            self.scene.dataset_type
        )
        
        hopacloss = ((1.0 - self.gaussians.get_hopac)**2).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()
        wopacloss = ((self.gaussians.get_wopac).abs()).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()

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


        planeloss = self.gaussians.compute_regulation(
            self.hyperparams.time_smoothness_weight, self.hyperparams.l1_time_planes, self.hyperparams.plane_tv_weight,
            self.hyperparams.minview_weight
        )

        loss = L1 +  planeloss + \
                        pg_loss + \
                            0.01*(hopacloss) + (wopacloss)
                   
        # print( planeloss ,depthloss,hopacloss ,wopacloss ,normloss ,pg_loss,covloss)
        with torch.no_grad():
            if self.gui:
                    dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                    dpg.set_value("_log_loss", f"Loss: {L1.item()}")
                    dpg.set_value("_log_opacs", f"h/w: {hopacloss}  | {wopacloss} ")
                    dpg.set_value("_log_depth", f"PhysG: {pg_loss} ")
                    dpg.set_value("_log_dynscales", f"Plane Reg: {planeloss} ")

                    if (self.iteration % 1000) == 0:
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
            self.timer.pause() # log and save
           
            torch.cuda.synchronize()
            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations) or (self.iteration == self.final_iter-1):
                self.save_scene()

            self.timer.start()

            if self.iteration % 100 == 0 and self.iteration < self.final_iter - 200:
                self.gaussians.compute_3D_filter(cameras=self.filter_3D_stack)
            
            if self.iteration % self.opt.opacity_reset_interval == 0 and self.iteration < 8001:# and self.stage == 'fine':
                self.gaussians.reset_opacity()

    def train_depth_step(self):

        # Start recording step duration
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.iter_start.record()

        zero_cam_idxs = self.scene.train_camera.zero_idxs
        
        
        
        C = self.scene.max_frames #300 if self.scene.dataset_type == 'dynerf' else 300
        
        canon_cams = []
        view_cams = []
        while view_cams == []:
            viewpoint_cams = self.get_batch_views
            # Sample the image at t=0 as the reference
            for cam in viewpoint_cams:
                if cam.uid % C != 0:
                    idx = zero_cam_idxs.index(cam.uid - (cam.uid % C))
                    canon_cams.append(self.coarse_viewpoint_stack[idx])
                    view_cams.append(cam)
        
        loss = render_depth_batch(
            view_cams, canon_cams,
            self.gaussians
        )
      
        with torch.no_grad():
            if self.gui:
                dpg.set_value("_log_knn", f"Depth: {loss.item()} | cov {0.} ")
       
            # Error if loss becomes nan
            if torch.isnan(loss).any():
                print("loss is nan for `gui.py:train_depth_step(...)`, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
        # Backpass
        loss.backward()
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        self.iter_end.record()
    
    
    @torch.no_grad()
    def full_evaluation(self):
        print('Full Eval')
        import lpips
        # from skimage.metrics import structural_similarity as ssim
        lpips_vgg = lpips.LPIPS(net='vgg').to('cuda')
        lpips_alex = lpips.LPIPS(net='alex').to('cuda')

        @torch.no_grad()
        def psnr(img1, img2, mask=None):
            if mask is not None:
                assert mask.shape == img1.shape[-2:], "Mask must match HxW of the image"
                mask = mask.expand_as(img1)
                diff = (img1 - img2) ** 2 * mask
                mse = diff.sum() / mask.sum()
            else:
                mse = ((img1 - img2) ** 2).mean()
            
            mse = torch.clamp(mse, min=1e-10)  # Prevent log(0)
            psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            return psnr_value

        self.test_viewpoint_stack = self.scene.getTestCameras() #.copy()
        viewpoint_cams = self.test_viewpoint_stack
        per_frame_results = {i: {'full':{'psnr': 0., 'ssim': 0., 'lpips_vgg': 0., 'lpips_alex': 0.},
                                 'mask':{'psnr': 0.,'m_mae':0.,'m_psnr': 0., 'ssim': 0., 'lpips_vgg': 0., 'lpips_alex': 0.}} for i in range(len(viewpoint_cams))}

        cnt = 0
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
                stage='test-full' #foreground
            )
            test_img = render_pkg["render"]
            gt_img = viewpoint_cam.original_image.cuda()
            mask = viewpoint_cam.gt_alpha_mask.cuda().unsqueeze(0)
            
            if self.scene.dataset_type == 'condense':
                A=59
                B=2500
                C=34
                D=1405
                
                test_img = test_img[:, A:B,C:D]
                gt_img = gt_img[:, A:B,C:D]
                mask = mask[:, A:B,C:D]
            
            save_gt_pred_full(test_img, cnt, self.args.expname)
            
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 5))  # Adjust figure size if needed
            # plt.subplot(1, 2, 1)  # Left side
            # plt.imshow(test_img.permute(1,2,0).cpu().detach().numpy(), cmap='gray')  # Use 'gray' if it's a grayscale image
            # plt.title("Test Image")
            # plt.axis('off')
            
            # plt.subplot(1, 2, 2)  # Right side
            # plt.imshow(gt_img.permute(1,2,0).cpu().detach().numpy(), cmap='gray')
            # plt.title("Ground Truth Image")
            # plt.axis('off')
            
            # plt.show()
            # exit()
            test_mask = test_img * mask
            gt_mask = gt_img * mask

            per_frame_results[cnt]['full']['psnr']  += psnr(gt_img, test_img).item()
            per_frame_results[cnt]['full']['ssim']  += ssim(gt_img, test_img, window_size=3).item()
            per_frame_results[cnt]['full']['lpips_vgg'] += lpips_vgg(gt_img, test_img).item()
            per_frame_results[cnt]['full']['lpips_alex'] += lpips_alex(gt_img, test_img).item()
            
            # Masked
            per_frame_results[cnt]['mask']['psnr']  += psnr(gt_mask, test_mask).item()
            per_frame_results[cnt]['mask']['m_psnr']  += psnr(gt_img, test_img, mask.squeeze(0)).item()
            per_frame_results[cnt]['mask']['ssim']  += ssim(gt_mask, test_mask, window_size=3).item()
            
            per_frame_results[cnt]['mask']['lpips_vgg'] += lpips_vgg(gt_mask, test_mask).item()
            per_frame_results[cnt]['mask']['lpips_alex'] += lpips_alex(gt_mask, test_mask).item()

            
            render_pkg = render(
                viewpoint_cam, 
                self.gaussians, 
                self.pipe, 
                self.background, 
                stage='test-foreground' #foreground
            )
            test_img = render_pkg["render"]
            save_gt_pred_mask(test_img, cnt, self.args.expname)

            cnt += 1

        average = {
            'full': {k: 0. for k in ['mae', 'psnr', 'ssim', 'lpips_vgg', 'lpips_alex']},
            'mask': {k: 0. for k in ['mae', 'psnr', 'm_mae', 'm_psnr', 'ssim', 'lpips_vgg', 'lpips_alex']}
        }

        # Accumulate
        for frame_data in per_frame_results.values():
            for category in ['full', 'mask']:
                for metric in average[category]:
                    average[category][metric] += frame_data[category].get(metric, 0.0)

        # Average
        num_frames = len(viewpoint_cams)
        for category in average:
            for metric in average[category]:
                average[category][metric] /= num_frames
                
        import json
        with open(f'output/{self.args.expname}/results.json', 'w') as json_file:
            json.dump({
                "average":average,
                "per-frame":per_frame_results
                }, json_file,  indent=4)
     
    @torch.no_grad()
    def test_step(self):
        @torch.no_grad()
        def psnr(img1, img2, mask=None):
            if mask is not None:
                assert mask.shape == img1.shape[-2:], "Mask must match HxW of the image"
                mask = mask.expand_as(img1)
                diff = (img1 - img2) ** 2 * mask
                mse = diff.sum() / mask.sum()
            else:
                mse = ((img1 - img2) ** 2).mean()
            
            mse = torch.clamp(mse, min=1e-10)  # Prevent log(0)
            psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            return psnr_value

        self.test_viewpoint_stack = self.scene.getTestCameras() #.copy()
        viewpoint_cams = self.test_viewpoint_stack
        per_frame_results = {i: {'full':{'psnr': 0., 'ssim': 0.},
                                 'mask':{'psnr': 0.,'m_psnr': 0., 'ssim': 0.}} for i in range(len(viewpoint_cams))}

        cnt = 0
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
                stage='test-full' #foreground
            )
            test_img = render_pkg["render"]
            gt_img = viewpoint_cam.original_image.cuda()
            mask = viewpoint_cam.gt_alpha_mask.cuda().unsqueeze(0)
            
            if self.scene.dataset_type == 'condense':
                A=59
                B=2500
                C=34
                D=1405
                
                test_img = test_img[:, A:B,C:D]
                gt_img = gt_img[:, A:B,C:D]
                mask = mask[:, A:B,C:D]
               
            test_mask = test_img * mask
            gt_mask = gt_img * mask

            per_frame_results[cnt]['full']['psnr']  += psnr(gt_img, test_img).item()
            per_frame_results[cnt]['full']['ssim']  += ssim(gt_img, test_img, window_size=3).item()
            # Masked
            per_frame_results[cnt]['mask']['psnr']  += psnr(gt_mask, test_mask).item()
            per_frame_results[cnt]['mask']['m_psnr']  += psnr(gt_img, test_img, mask.squeeze(0)).item()
            per_frame_results[cnt]['mask']['ssim']  += ssim(gt_mask, test_mask, window_size=3).item()
            
            cnt += 1

        average = {
            'full': {k: 0. for k in ['psnr', 'ssim']},
            'mask': {k: 0. for k in ['psnr', 'm_psnr', 'ssim']}
        }

        # Accumulate
        for frame_data in per_frame_results.values():
            for category in ['full', 'mask']:
                for metric in average[category]:
                    average[category][metric] += frame_data[category].get(metric, 0.0)

        # Average
        num_frames = len(viewpoint_cams)
        for category in average:
            for metric in average[category]:
                average[category][metric] /= num_frames
        print(f'Test {self.iteration} | {average}')
        
    @torch.no_grad()
    def render_video_step(self):

        viewpoint_cams = self.scene.getVideoCameras() #.copy()

        cnt = 0
        # Render and return preds
        for viewpoint_cam in tqdm(viewpoint_cams, desc="Processing viewpoints"):
            save_novel_views(render(
                viewpoint_cam, 
                self.gaussians, 
                self.pipe, 
                self.background, 
                stage='test-full'
            )["render"], cnt, self.args.expname)
            save_novel_views_foreground(render(
                viewpoint_cam, 
                self.gaussians, 
                self.pipe, 
                self.background, 
                stage='test-foreground'
            )["render"], cnt, self.args.expname)
            
            cnt += 1


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def save_novel_views(pred, idx, name):

    pred = (pred.permute(1, 2, 0)
        # .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )*255

    pred = pred.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    if not os.path.exists(f'output/{name}/nvs/'):
        os.mkdir(f'output/{name}/nvs/')
        os.mkdir(f'output/{name}/nvs/full/')
    elif not os.path.exists(f'output/{name}/nvs/full/'):
        os.mkdir(f'output/{name}/nvs/full/')
    cv2.imwrite(f'output/{name}/nvs/full/{idx}.png', pred_bgr)

    return pred_bgr
def save_novel_views_foreground(pred, idx, name):

    pred = (pred.permute(1, 2, 0)
        # .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )*255

    pred = pred.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    if not os.path.exists(f'output/{name}/nvs/'):
        os.mkdir(f'output/{name}/nvs/')
        os.mkdir(f'output/{name}/nvs/foreground/')
    elif not os.path.exists(f'output/{name}/nvs/foreground/'):
        os.mkdir(f'output/{name}/nvs/foreground/')
    cv2.imwrite(f'output/{name}/nvs/foreground/{idx}.png', pred_bgr)

    return pred_bgr

def save_gt_pred_mask(pred, idx, name):

    pred = (pred.permute(1, 2, 0)
        # .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )*255

    pred = pred.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    if not os.path.exists(f'output/{name}/foreground/'):
        os.mkdir(f'output/{name}/foreground/')
    cv2.imwrite(f'output/{name}/foreground/{idx}.png', pred_bgr)

    return pred_bgr

def save_gt_pred_full(pred, idx, name):

    pred = (pred.permute(1, 2, 0)
        # .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )*255

    pred = pred.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    if not os.path.exists(f'output/{name}/images/'):
        os.mkdir(f'output/{name}/images/')
    cv2.imwrite(f'output/{name}/images/{idx}.png', pred_bgr)

    return pred_bgr

def save_video(pred, idx, name):
    path =  os.path.join('output',name, 'render')
    if os.path.exists(path) == False:
        os.makedirs(path)
    
    pred = (pred.permute(1, 2, 0)
        # .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )*255

    pred = pred.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    dest = os.path.join(path, f'{int(idx):03d}.png')
    cv2.imwrite(dest, pred_bgr)

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
    parser.add_argument("--test_iterations", type=int, default=4000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[8000, 15999, 20000, 30_000, 45000, 60000])
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
        view_test=args.view_test,
        use_gui=True
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