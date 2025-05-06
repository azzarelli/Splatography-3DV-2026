
import shutil
import dearpygui.dearpygui as dpg
import numpy as np
import random
import os, sys
import torch
from random import randint
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

from gaussian_renderer import render,render_batch
import json
import open3d as o3d
from submodules.DAV2.depth_anything_v2.dpt import DepthAnythingV2
from gui_utils.base import get_in_view_dyn_mask


def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask
    
def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

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

        self.total_frames = 50

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
            self.random_loader = True
            # if self.stage == 'fine':
            print('Loading Fine (t = any) dataset')
            self.scene.getTrainCameras().dataset.get_mask = True

            self.viewpoint_stack = self.scene.getTrainCameras() # [self.scene.index_train(i*50) for i in range(4*self.total_frames)] # self.scene.getTrainCameras()
            
            self.loader = iter(DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                num_workers=32, collate_fn=list))
            self.scene.getTrainCameras().dataset.get_mask = False
            # if self.stage == 'coarse': 
            #     print('Loading Coarse (t=0) dataset')
            #     self.scene.getTrainCameras().dataset.get_mask = True
            #     self.viewpoint_stack = [[self.scene.index_train((i*50)) for i in range(4)]] * 100 # self.scene.getTrainCameras()
            #     self.scene.getTrainCameras().dataset.get_mask = False

            #     self.random_loader  = True
            #     self.loader = iter(DataLoader(self.viewpoint_stack, batch_size=1, shuffle=self.random_loader,
            #                                         num_workers=32, collate_fn=list))

    @property
    def get_zero_cams(self):
        self.scene.getTrainCameras().dataset.get_mask = True
        zero_cams = [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]
        self.scene.getTrainCameras().dataset.get_mask = False
        return zero_cams

      
    
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
            
        # Handle Data Loading:
        try:
            viewpoint_cams = next(self.loader)
        except StopIteration:
            viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=self.random_loader,
                                                num_workers=32, collate_fn=list)
            self.loader = iter(viewpoint_stack_loader)
            viewpoint_cams = next(self.loader)

        # Render
        if (self.iteration - 1) == self.debug_from:
            self.pipe.debug = True

        # Generate scene based on an input camera from our current batch (defined by viewpoint_cams)
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        L1 = torch.tensor(0.).cuda()
        dyn_target_loss = torch.tensor(0.).cuda()
        dyn_scale_loss= torch.tensor(0.).cuda()
        opacloss = torch.tensor(0.).cuda()
        pg_loss = torch.tensor(0.).cuda()
        depth_loss = torch.tensor(0.).cuda()
        
        radii_list, visibility_filter_list, viewspace_point_tensor_list, L1, extras = render_batch(
            viewpoint_cams, 
            self.gaussians, 
            self.pipe,
            self.background, 
            stage=self.stage,
        )
        
        # Global regularization          
        opacloss += ((1.0 - self.gaussians.get_hopac)**2).mean()  #+ ((self.gaussians.get_h_opacity[self.gaussians.get_h_opacity < 0.2])**2).mean()

        # Regularizers for fine stage
        if self.stage == 'fine':
            L1 += self.gaussians.compute_regulation(
                self.hyperparams.time_smoothness_weight, 
                self.hyperparams.l1_time_planes, 
                self.hyperparams.plane_tv_weight,
            )
            # dyn_target_loss += self.gaussians.compute_rigidity_loss()
        else:
            pass
            # dyn_target_loss += self.gaussians.compute_static_rigidity_loss()
            # scale_exp = self.gaussians.get_scaling
            # opac_int = self.gaussians.opacity_integral(w=w_, h=h_, mu=mu_)
            # pg_loss = 0.01* (scale_exp.max(dim=1).values / scale_exp.min(dim=1).values).mean()
            
        # max_gauss_ratio = 10
        # pg_loss = (
        #     torch.maximum(
        #         scale_exp.amax(dim=-1)  / scale_exp.amin(dim=-1),
        #         torch.tensor(max_gauss_ratio),
        #     )
        #     - max_gauss_ratio
        # ).mean()

        loss = L1 + pg_loss + dyn_scale_loss + opacloss + dyn_target_loss

        
        with torch.no_grad():
            if self.gui:
                    dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                    dpg.set_value("_log_loss", f"Loss: {loss.item()} ")
                    dpg.set_value("_log_opacs", f"Opac loss: {opacloss} ")
                    dpg.set_value("_log_depth", f"Depth loss: {depth_loss} ")
                    dpg.set_value("_log_dynscales", f"DynScales loss: {dyn_scale_loss} ")
                    dpg.set_value("_log_knn", f"Target loss: {dyn_target_loss} ")
                    if (self.iteration % 2) == 0:
                        dpg.set_value("_log_points", f"Scene Pts: {self.gaussians._xyz.shape[0]}")
                    

            if self.iteration % 2000 == 0:
                self.track_cpu_gpu_usage(0.1)
            # Error if loss becomes nan
            if torch.isnan(loss).any():
                if torch.isnan(pg_loss):
                    print("PG loss is nan!")
                if torch.isnan(opacloss):
                    print("Opac loss is nan!")
                if torch.isnan(dyn_scale_loss):
                    print("Dyn loss is nan!")
                if torch.isnan(dyn_target_loss):
                    print("Dyn target loss is nan!")
                        
                # if torch.isnan(dyn_scale_loss):
                #     print("Dyn Scale is nan!")
                    
                print("loss is nan, end training, reexecv program now.")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
        # Backpass
        loss.backward()
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        self.iter_end.record()
        
        with torch.no_grad():
            
            radii = torch.cat(radii_list, 0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[0])
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            
            self.timer.pause() # log and save
           
            torch.cuda.synchronize()
            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations) or (self.stage == 'coarse' and self.iteration == 2999):
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration, self.stage)
    
                print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
                torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + f"_{self.stage}_" + str(self.iteration) + ".pth")
    

            self.timer.start()
            
            # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

            # Densification for scene
            densify_threshold = self.opt.densify_grad_threshold
            if  self.stage == 'fine' and \
                self.iteration > self.opt.densify_from_iter and \
                self.iteration < self.opt.densify_until_iter and \
                self.iteration % self.opt.densification_interval == 0 and self.gaussians._xyz.shape[0] < 300000:
                    self.gaussians.densify(densify_threshold,self.scene.cameras_extent)

            # Global pruning
            if  self.stage == 'fine' and \
                self.iteration > self.opt.pruning_from_iter and \
                self.iteration % self.opt.pruning_interval == 0 and self.gaussians._xyz.shape[0] > 2:
                size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                self.gaussians.prune(self.hyperparams.opacity_lambda, self.scene.cameras_extent, size_threshold)
            
    
            if self.iteration % self.opt.opacity_reset_interval == 0 and self.stage == 'fine':
                self.gaussians.reset_opacity()
                
            # Optimizer step
            # if self.iteration  == 3000:
            #     self.gaussians.densify_target(zero_cams)

            
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True)
            
    @torch.no_grad()
    def test_step(self):

        if self.iteration < (self.final_iter -1) and (self.iteration % 500) != 0:
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
            viewpoint_cams = self.test_viewpoint_stack


        # if not os.path.exists(f'debugging/{self.iteration}') and self.iteration % 1000 == 0:
        #     os.mkdir(f'debugging/{self.iteration}')

        PSNR = 0.
        SSIM = 0.

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
                stage=self.stage
            )
            image = render_pkg["render"]


            gt_image = viewpoint_cam.original_image.cuda()


            if (self.iteration == 500) or (self.iteration == 1000 or self.iteration == 2000):
                if idx % 3 == 0:
                    save_gt_pred(gt_image, image, self.iteration, idx, self.args.expname.split('/')[-1])

            # mask = self.dilation_transform(viewpoint_cam.mask.cuda())

            try:
                PSNR += psnr(image, gt_image, viewpoint_cam.mask.cuda())
            except:
                PSNR += psnr(image, gt_image)
            SSIM += ssim(image.unsqueeze(0), gt_image)
            idx += 1

            if idx % 4 == 0 and self.gui:
                with torch.no_grad():
                    self.viewer_step()
                    dpg.render_dearpygui_frame()


        # Loss
        PSNR = PSNR.item()/len(viewpoint_cams)
        SSIM = SSIM/len(viewpoint_cams)

        save_file = os.path.join(self.results_dir, f'{self.iteration}.json')
        with open(save_file, 'w') as f:
            obj = {
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


def pairwise_distances_func(gt, pred):
    # Shapes: gt -> (N, 3), pred -> (M, 3)
    gt_norm = torch.sum(gt**2, dim=1).unsqueeze(1)  # Shape: (N, 1)
    pred_norm = torch.sum(pred**2, dim=1).unsqueeze(0)  # Shape: (1, M)
    cross_term = torch.matmul(gt, pred.T)  # Shape: (N, M)
    distances = torch.sqrt(gt_norm - 2 * cross_term + pred_norm)  # Shape: (N, M)
    return distances

def custom_pcd_loss(gt, pred, k= 4, threshold=0.01):
    gt = torch.tensor(o3d.io.read_point_cloud(gt).points).float().cuda()
    stability = 0.0000001
    # print(pred.dtype, gt.dtype)
    if gt.shape[0] < 5000:
        # TODO: Decide on point-based regularisation
        diff = gt[:, None, :] - pred[None, :, :]  # Shape: (N, M, 3)
        pairwise_distances = torch.norm(diff, dim=2)  # Shape: (N, M)
        distances, indices = torch.topk(pairwise_distances, k, dim=1, largest=False, sorted=True)
        return F.relu(distances.mean(-1) - threshold).mean()

    return None

def nns(knn_indices, features):
    # Gather neighbor features without for-loop
    safe_indices = knn_indices.clone()
    safe_indices[safe_indices == -1] = 0  # avoid indexing error
    neighbor_features = features[safe_indices]  # (N, k, F)

    # Mask out invalid neighbors
    valid_mask = (knn_indices != -1).unsqueeze(-1)  # (N, k, 1)
    neighbor_features = neighbor_features * valid_mask

    # Mean of valid neighbor features
    neighbor_sum = neighbor_features.sum(dim=1)
    neighbor_count = valid_mask.sum(dim=1).clamp(min=1)
    neighbor_mean = neighbor_sum / neighbor_count
    
    return neighbor_mean

# from pytorch3d.ops import ball_query, knn_gather

# def KNN_motion_features(positions, features):
#     N, F = features.shape
#     k = 4
#     radius = 2
    
#     N, _ = positions.shape

#     # PyTorch3D expects batched inputs
#     device = positions.device
#     N, _ = positions.shape

#     # Add batch dimension
#     pos = positions.unsqueeze(0)  # (1, N, 3)
#     feat = features.unsqueeze(0)  # (1, N, F)

#     # Get K nearest neighbors
#     knn = knn_points(pos, pos, K=k+1)  # includes self as neighbor

#     # Exclude the first neighbor (self)
#     idx = knn.idx[:, :, 1:]  # (1, N, k)
#     neighbor_feats = knn_gather(feat, idx)  # (1, N, k, F)

#     # Compute mean of neighbors
#     neighbor_mean = neighbor_feats.mean(dim=2)  # (1, N, F)

#     # Difference between feature and neighborhood mean
#     diff = feat - neighbor_mean  # (1, N, F)
#     return (diff**2).mean()


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

    image_array = np.hstack((gt, pred))

    # image_array = image_array.astype(np.uint8)

    cv2.imwrite(f'debugging/{iteration}/{name}_{idx}.png', image_array)

    return image_array

SQRT_PI = torch.sqrt(torch.tensor(torch.pi))
def gaussian_integral(h, w, mu):
    """Returns high weight (0 to 1) for solid materials
    
        Notes on optimization:
            We evaluate the function return 1 - (SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2)
            However, as this tends to 0 when out point is solid, we will do:
            return 1- (1 - (SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2)) = (SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2)
 
            Note, we may want to remove transparent materials and this can be done by 
            h*(SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2), which returns a low weight for transparent materiasl
    """
    erf_term_1 = torch.erf(w * mu)
    erf_term_2 = torch.erf(w * (mu - 1))
    return ((SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2)).squeeze(-1)

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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 9999,15999, 20000, 30_000, 45000, 60000])
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

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(True)
    gui = GUI(
        args=args, 
        hyperparams=hp.extract(args), 
        dataset=lp.extract(args), 
        opt=op.extract(args), 
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations, 
        saving_iterations=args.save_iterations,
        ckpt_start=args.start_checkpoint, 
        debug_from=args.debug_from, 
        expname=args.expname,
        skip_coarse=args.skip_coarse,
        view_test=args.view_test
    )

    
    gui.render()

    print("\nTraining complete.")