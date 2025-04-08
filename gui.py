
import math
import shutil

import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R


import numpy as np
import random
import os, sys
import torch
from random import randint
from tqdm import tqdm
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from pytorch_msssim import ms_ssim
import cv2

from utils.scene_utils import render_training_image
import copy
from gaussian_renderer import render, render_no_train, deform_gs
import json
from scene.condense_dataset import DilationTransform

import open3d as o3d

import psutil

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, time, from_training_view=False):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.time = time

        
        loaded = False
        if from_training_view:
            if type(c2w) == type({'type':'dict'}):
                self.world_view_transform = c2w['world_view_transform']

                self.c2w = torch.linalg.inv(self.world_view_transform.cuda().transpose(0,1))
                self.projection_matrix = c2w['projection_matrix']
                self.full_proj_transform = c2w['full_proj_transform']

                loaded = True
        
        if not loaded:
            self.c2w = c2w
            w2c = np.linalg.inv(c2w)
            # rectify...
            w2c[1:3, :3] *= -1
            w2c[:3, 3] *= -1

            self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
            self.projection_matrix = (
                getProjectionMatrix(
                    znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
                )
                .transpose(0, 1)
                .cuda().float()
            )
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        self.camera_center = -torch.tensor(self.c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

 
class OrbitCamera:
    def __init__(self, W, H, r=2, fov=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        
        self.fovy = 2 * np.arctan(np.tan(fov[0]/2)* 0.25)
        self.fovy = np.deg2rad(self.fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.array([[1., 0., 0.,],
                                           [0., 0., -1.],
                                           [0., 1., 0.]]))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.side = np.array([1, 0, 0], dtype=np.float32)

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        up = self.rot.as_matrix()[:3, 1]
        rotvec_x = up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0001 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

    def get_proj_matrix(self):
        tanHalfFovY = math.tan((self.fovy  / 2))
        tanHalfFovX = math.tan((self.fovx / 2))

        top = tanHalfFovY * self.near
        bottom = -top
        right = tanHalfFovX * self.near
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * self.near / (right - left)
        P[1, 1] = 2.0 * self.near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.far / (self.far - self.near)
        P[2, 3] = -(self.far * self.near) / (self.far - self.near)
        return P

# Going from 3x3 rotation matrix to quaternion values
def R_to_q(R,eps=1e-8): # [B,3,3]
            # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
            # FIXME: this function seems a bit problematic, need to double-check
            row0,row1,row2 = R.unbind(dim=-2)
            R00,R01,R02 = row0.unbind(dim=-1)
            R10,R11,R12 = row1.unbind(dim=-1)
            R20,R21,R22 = row2.unbind(dim=-1)
            t = R[...,0,0]+R[...,1,1]+R[...,2,2]
            r = (1+t+eps).sqrt()
            qa = 0.5*r
            qb = (R21-R12).sign()*0.5*(1+R00-R11-R22+eps).sqrt()
            qc = (R02-R20).sign()*0.5*(1-R00+R11-R22+eps).sqrt()
            qd = (R10-R01).sign()*0.5*(1-R00-R11+R22+eps).sqrt()
            q = torch.stack([qa,qb,qc,qd],dim=-1)
            
            for i,qi in enumerate(q):
                if torch.isnan(qi).any():
                    K = torch.stack([torch.stack([R00-R11-R22,R10+R01,R20+R02,R12-R21],dim=-1),
                                    torch.stack([R10+R01,R11-R00-R22,R21+R12,R20-R02],dim=-1),
                                    torch.stack([R20+R02,R21+R12,R22-R00-R11,R01-R10],dim=-1),
                                    torch.stack([R12-R21,R20-R02,R01-R10,R00+R11+R22],dim=-1)],dim=-2)/3.0
                    K = K[i]
                    eigval,eigvec = torch.linalg.eigh(K)
                    V = eigvec[:,eigval.argmax()]
                    q[i] = torch.stack([V[3],V[0],V[1],V[2]])
            return q

def hash_cams(cam):
    return cam.camera_center.sum().item()

class GaussianCameraModel:

    def get_camera_stuff(self, cam_list):
        xyzs = []
        qs = []
        
        hash_camlist = []

        # For each camera in list get the camera position and rotation
        for i, cam in enumerate(cam_list):
            # Hash the camera (i.e. avoid displaying cameras with the same R and T)
            cam_hash = hash_cams(cam)
            # We do this but summing the camera center points (sure it seems silly but this is realistically fine....(I think)
            if cam_hash not in hash_camlist:
                hash_camlist.append(cam_hash)

                for xyz in self.cam_xyzs:
                    T = cam.camera_center + xyz.cpu()
                    w2c = cam.world_view_transform.transpose(0,1)
                    R = w2c[:3, :3].transpose(0,1)
                    T = torch.matmul(R, xyz.unsqueeze(-1).cpu()).squeeze(-1) + cam.camera_center

                    q = R_to_q(R)

                    xyzs.append(T.unsqueeze(0))
                    qs.append(q.unsqueeze(0))
            # else:
            #     break

        self.qs = torch.cat(qs, dim=0).cuda()
        self.xyzs = torch.cat(xyzs, dim=0).cuda()

        self.xyzs.requires_grad = True
        self.qs.requires_grad = True

    def get_cam_model(self, cam, W, H):
        w = float(W/2.)
        h = float(H/2.)
        
        x_num_pts = 11

        w2c = cam.world_view_transform.transpose(0,1)
        R = w2c[:3, :3].transpose(0,1)
        xyzs = []
        qs = []

        if w > h:
            x_size = float(4.*x_num_pts)
            y_size = float(3.*x_num_pts)
        elif w < h:
            x_size = float(3.*x_num_pts)
            y_size = float(4.*x_num_pts)
        else:
            x_size = float(3.*x_num_pts)
            y_size = float(3.*x_num_pts)
            
        # Line in +X (Right) 
        for i in range(0, x_num_pts+1):
            x = float(i-(x_num_pts)/2.)/x_size
            T = torch.tensor([x, +0.2, 0.])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))

        for i in range(0, x_num_pts+1):
            x = float(i-(x_num_pts)/2.)/x_size
            T = torch.tensor([x, -0.2, 0.])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))
        
        # Line in -Z (Back)
        for i in range(0, 5):
            T = torch.tensor([0., 0., - float(i/10.)])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))

        # Line in +Y (Down) Show direction
        T = torch.tensor([-0.1, 0., 0.])
        q = R_to_q(R)
        xyzs.append(T.unsqueeze(0))
        qs.append(q.unsqueeze(0))

        # Show -x -y for the topleft hand image
        T = torch.tensor([0., -0.1, 0.])
        q = R_to_q(R)
        xyzs.append(T.unsqueeze(0))
        qs.append(q.unsqueeze(0))
        
        
        self.cam_qs = torch.cat(qs, dim=0)
        self.cam_xyzs = torch.cat(xyzs, dim=0)

    def __init__(self, cam_list, W, H) -> None:

        max_i = 1
        if len(cam_list) > 100: max_i = 300

        try:
            cam = cam_list[0][0]
            cam_list = [cam_list[i][0] for i in range(len(cam_list)) if i % max_i == 0]
        except:
            cam = cam_list[0]
            cam_list = [cam_list[i] for i in range(len(cam_list)) if i % max_i == 0]
        self.get_cam_model(cam, W, H)

        self.get_camera_stuff(cam_list)

        cam_centers = []
        cam_hash = []

        for i, cam in enumerate(cam_list):
            T = cam.camera_center
            if str(T) not in cam_hash:
                cam_centers.append(T)
                cam_hash.append(str(T))
        tensor_stack = torch.stack(cam_centers)
        self.cam_center = torch.mean(tensor_stack, dim=0)
        
        # Blob scale NOT camera scale
        self.scale = torch.tensor([0.001, 0.001, 0.001]).cuda()


class GUI:

    def __init__(self, 
                 args, 
                 hyperparams, 
                 dataset, 
                 opt, 
                 pipe, 
                 testing_iterations, 
                 saving_iterations,
                 ckpt_it,
                 ckpt_start,
                 debug_from,
                 expname
                 ):
        self.gui = True

        self.dataset = dataset
        self.hyperparams = hyperparams
        self.args = args

        self.expname = expname

        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations

        self.saving_iterations = saving_iterations

        self.checkpoint_iterations = ckpt_it
        self.checkpoint = ckpt_start
        self.debug_from = debug_from

        if self.dataset.white_background:
            bg_color = [1, 1, 1]
        else:
            bg_color = [0,0,0]
        bg_color = [0, 0, 0]

        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.tb_writer = prepare_output_and_logger(expname)
        self.gaussians = GaussianModel(dataset.sh_degree, hyperparams)
        dataset.model_path = args.model_path

        self.timer = Timer()
        self.scene = Scene(dataset, self.gaussians)
        self.timer.start()

        self.stage = 'coarse'

        self.init_taining()

        try:
            self.W, self.H = self.scene.getTestCameras()[0][0].image_width, self.scene.getTestCameras()[0][0].image_height
            self.fov = (self.scene.getTestCameras()[0][0].FoVy, self.scene.getTestCameras()[0][0].FoVx)
        except:
            self.W, self.H = self.scene.getTestCameras()[0].image_width, self.scene.getTestCameras()[0].image_height
            self.fov = (self.scene.getTestCameras()[0].FoVy, self.scene.getTestCameras()[0].FoVx)

        self.W = int(self.W/2)
        self.H = int(self.H/2)

        self.fovy = self.fov[0]
        self.cam = OrbitCamera(self.W, self.H, r=60., fov=self.fov)
        self.store_orbit = self.cam
        self.mode = "rgb"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.time = 0.
        self.show_radius = 30.
        self.show_scene = True
        self.show_depth = False
        self.show_cameras = True
        self.show_test = False

        self.results_dir = os.path.join(self.args.model_path, 'active_results')
        if os.path.exists(self.results_dir):
            print(f'[Removing old results] : {self.results_dir}')
            shutil.rmtree(self.results_dir)
        os.mkdir(self.results_dir)
        self.training_cams_pc = GaussianCameraModel(self.scene.getTrainCameras(), self.W, self.H)

        # For evaluating masked images with psnr
        # self.dilation_transform = DilationTransform() # For image assessment involving masks

        if self.gui:
            print('DPG loading ...')
            dpg.create_context()
            self.register_dpg()

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
            if 'fine' in self.checkpoint:
                (model_params, first_iter) = torch.load(self.checkpoint)
                self.gaussians.restore(model_params, self.opt)

        # Set current iteration
        self.iteration = first_iter

        # Events for counting duration of step
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)

        if self.stage == 'coarse':
            self.viewpoint_stack = self.scene.getTrainCameras()
            self.test_viewpoint_stack = self.scene.getTestCameras()

            self.random_loader  = True
            self.loader = iter(DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=True,
                                                num_workers=16, collate_fn=list))

        self.load_in_memory = False


    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=400,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            with dpg.group():
                dpg.add_text("Training info: ")
                dpg.add_text("no data", tag="_log_iter")
                dpg.add_text("no data", tag="_log_loss")
                dpg.add_text("no data", tag="_log_depth")
                dpg.add_text("no data", tag="_log_points")

            with dpg.collapsing_header(label="Testing info:", default_open=True):
                dpg.add_text("no data", tag="_log_psnr_test")
                dpg.add_text("no data", tag="_log_ssim")


            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                
                def callback_toggle_show_depth(sender):
                    self.show_depth = not self.show_depth
                    
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Depth", callback=callback_toggle_show_depth)
                    
                def callback_show_max_radius(sender):
                    self.show_radius = dpg.get_value(sender)
                    self.need_update = True      
                    
                dpg.add_slider_float(
                    label="Show Radial Distance",
                    default_value=30.,
                    max_value=50.,
                    min_value=0.,
                    callback=callback_show_max_radius,
                )
                
                def callback_speed_control(sender):
                    self.time = dpg.get_value(sender)
                    self.need_update = True                    
                    
                dpg.add_slider_float(
                    label="Time",
                    default_value=0.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_speed_control,
                )

                def callback_toggle_show_scene(sender):
                    self.show_scene = not self.show_scene

                def callback_toggle_show_cameras(sender):
                    self.show_cameras = not self.show_cameras
                    
                def callback_toggle_show_tes(sender):
                    self.show_test = not self.show_test

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Show/Hide Scene", callback=callback_toggle_show_scene)
                    dpg.add_button(label="Show/Hide Cameras", callback=callback_toggle_show_cameras)
                    dpg.add_button(label="Show/Hide Test", callback=callback_toggle_show_tes)



        
        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

                
            self.cam.orbit(dx, dy)
                
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data
            
            self.cam.scale(delta)

            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)

        dpg.create_viewport(
            title="WavePlanes",
            width=self.W + 400,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()

    def track_cpu_gpu_usage(self, time):
        # Print GPU and CPU memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 ** 2)  # Convert to MB

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        print(
            f'[{self.stage} {self.iteration}] Time: {time:.2f} | Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB | CPU Memory Usage: {memory_mb:.2f} MB')


    def render(self):

        if self.gui:
            while dpg.is_dearpygui_running():
                if self.iteration > self.final_iter and self.stage == 'coarse':
                    self.stage = 'fine'
                    self.init_taining()

                if self.iteration <= self.final_iter:
                    self.train_step()
                    self.iteration += 1


                if (self.iteration % self.args.test_iterations) == 0 or (self.iteration == 1 and self.stage == 'fine' and self.opt.coarse_iterations > 50):
                    if self.stage == 'fine':
                        self.test_step()

                if self.iteration > self.final_iter and self.stage == 'fine':
                    self.stage = 'done'
                    exit()

                with torch.no_grad():
                    self.viewer_step()
                    dpg.render_dearpygui_frame()
        else:
            while self.stage != 'done':
                if self.iteration % 100 == 0:
                    print(f'[{self.stage}] {self.iteration}')
                if self.iteration > self.final_iter and self.stage == 'coarse':
                    self.stage = 'fine'
                    self.init_taining()

                if self.iteration <= self.final_iter:
                    self.train_step()
                    self.iteration += 1


                if (self.iteration % self.args.test_iterations) == 0 or (self.iteration == 1 and self.stage == 'fine' and self.opt.coarse_iterations > 50):
                    if self.stage == 'fine':
                        self.test_step()

                if self.iteration > self.final_iter and self.stage == 'fine':
                    self.stage = 'done'
                    exit()

    @torch.no_grad()
    def viewer_step(self):
        
        if self.show_test:
            camera =copy.deepcopy(self.scene.getTestCameras()[0])
            camera.time = self.time
        else:
            camera = MiniCam(
                self.cam.pose,
                self.cam.W,
                self.cam.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far, 
                time=self.time)
        
        if not self.show_depth:
            tag = 'render'
        else:
            tag = 'depth'
        
        
            
        buffer_image = render_no_train(camera, 
                                    self.gaussians, 
                                    self.pipe, 
                                    self.background, 
                                    stage='fine', 
                                    cam_type=self.scene.dataset_type,
                                    cams_pc = {
                                        "xyzs": self.training_cams_pc.xyzs,
                                        "qs":self.training_cams_pc.qs,
                                        "scale":self.training_cams_pc.scale,
                                        "show_scene":self.show_scene,
                                        "show_cameras":self.show_cameras
                                    },
                                    show_radius=self.show_radius
                                    )[tag]
           
        if not self.show_depth:
            pass
        else:

            buffer_image = (buffer_image - buffer_image.min())/(buffer_image.max() - buffer_image.min())
            buffer_image = buffer_image.repeat(3,1,1)
        
        
        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H,self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        # ender.record()
        self.need_update = True

        # t = starter.elapsed_time(ender)

        buffer_image = self.buffer_image
        # dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
        dpg.set_value(
            "_texture", buffer_image
        )  # buffer must be contiguous, else seg fault!


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
        if self.opt.dataloader and not self.load_in_memory: # and self.stage == 'fine':
            try:
                viewpoint_cams = next(self.loader)
            except StopIteration:
                viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size, shuffle=True,
                                                    num_workers=16, collate_fn=list)
                self.random_loader = True
                self.loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(self.loader)
        else:
            viewpoint_cams = []            
            
            if not self.viewpoint_stack:
                    self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
            zero_idxs = self.scene.train_camera.zero_idxs
            
            index = random.randrange(len(zero_idxs))  # get random index
            item = self.viewpoint_stack[zero_idxs[index]]
            viewpoint_cams.append(item)

            while len(viewpoint_cams) < self.opt.batch_size :
                curr_idx = random.randrange(len(zero_idxs))
                if curr_idx != index:
                    item = self.viewpoint_stack[zero_idxs[index]]
                    viewpoint_cams.append(item)
                    
            # # Construct batch for current step
            # while idx < self.opt.batch_size :
            
            #     if len(viewpoint_cams) > 0:
            #         # Ensure we select a camera with a different view
            #         nodiff = True
            #         while nodiff:
            #             # Select a random frame and check if the global position T appears already in the batch
            #             idx = randint(0, len(self.viewpoint_stack) - 1)
            #             temp_T = self.viewpoint_stack[idx].T
            #             cnt = 0
            #             for v in viewpoint_cams:
            #                 if v.T == temp_T: cnt += 1
            #             # if it doesn't appear add it to the stack and break from the loop
            #             if cnt == 0:
            #                 nodiff = False
            #                 viewpoint_cam = self.viewpoint_stack.pop(idx)
            #     else:
            #         viewpoint_cam = random.choice(self.viewpoint_stack)
                    
            #     if viewpoint_cam.time == 0.0:
            #         viewpoint_cams.append(viewpoint_cam)
            #         idx +=1

        
        
        # Render
        if (self.iteration - 1) == self.debug_from:
            self.pipe.debug = True

        # Generate scene based on an input camera from our current batch (defined by viewpoint_cams)
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        L1 = 0.

        for viewpoint_cam in viewpoint_cams:
            try: # If we have seperate depth
                viewpoint_cam, pcd_path = viewpoint_cam
            except:
                pcd_path = None

            if pcd_path is not None:
                # self.track_cpu_gpu_usage(viewpoint_cam.time)

                # depth_loss += self.train_depth_step(pcd_path, viewpoint_cam.time)
                # self.track_cpu_gpu_usage(viewpoint_cam.time)
                pass

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, stage=self.stage, cam_type=self.scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if self.scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()

            # train the gaussians inside the mask
            L1 += l1_loss(image, gt_image, viewpoint_cam.mask)

            
            # gt_images.append(gt_image.unsqueeze(0))
            images.append(image.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)

        # Loss
        loss = .0
        

        if self.iteration % 1000 == 0:
            self.track_cpu_gpu_usage(viewpoint_cam.time)

        if self.hyperparams.time_smoothness_weight != 0 and self.stage == 'fine':
            loss += self.gaussians.compute_regulation(
                self.hyperparams.time_smoothness_weight, 
                self.hyperparams.l1_time_planes, 
                self.hyperparams.plane_tv_weight,
                )
            
            w_, h_, mu_ = self.gaussians.get_opacity
            
            # if self.iteration > 3000:
            loss += ((1 - h_)).mean()
            
            # if render_pkg['stfeats'] is not None:
            #     loss += 0.001 * KNN_motion_features(render_pkg['means3D'], render_pkg['stfeats'])

            

        if self.opt.lambda_dssim != 0:
            loss += self.opt.lambda_dssim * (1.0- ssim(torch.cat(images, 0),torch.cat(gt_images, 0)))
            loss += L1 # (1. - self.opt.lambda_dssim ) * L1 
        else:
            loss += L1
        
        # NeuSG flat Gaussian loss
        max_gauss_ratio = 5
        scale_exp = self.gaussians.get_scaling
        
        # if self.stage == 'fine':
        #     w = 0.1 * self.gaussians.get_dynamic_point_prob()
        # else:
        w = 0.1
        # loss += (w * torch.abs(scale_exp.amin(dim=-1))).mean()
        
        # Phys Gaussian Loss (as per nerfstudio implementaton)
        # loss += 0.1*(
        #     torch.maximum(
        #         scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
        #         torch.tensor(max_gauss_ratio),
        #     )
        #     - max_gauss_ratio
        # ).mean()
        s_v, _ = torch.topk(scale_exp, k=2, dim=-1)
        loss += w*(
            torch.maximum(
                s_v[:, 0] / s_v[:, 1],
                torch.tensor(max_gauss_ratio),
            )
            - max_gauss_ratio
        ).mean()

        

        # final regularizer - we want to model surface position differences for solid dynamic points (so occurs later in training)
        # if self.iteration > 1000 and self.stage == 'fine':
        #     loss += self.gaussians.compute_rigidity_loss()

        # Include depth loss:
        # loss = loss # + (depth_loss / len(viewpoint_cams))
        # Backpass
        loss.backward()

        # Error if loss becomes nan
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        
        # Record end of step
        self.iter_end.record()

        # Log and save
        with torch.no_grad():
            self.timer.pause() # log and save
            if self.gui:
                dpg.set_value("_log_iter", f"{self.iteration} / {self.final_iter} its")
                dpg.set_value("_log_loss", f"Loss: {loss.item()} ")

            if (self.iteration % 2) == 0 and self.gui:
                total_point = self.gaussians._xyz.shape[0]
                dpg.set_value("_log_points", f"{total_point} total points")

            torch.cuda.synchronize()

            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration, self.stage)

            self.timer.start()
            
            # Densification
            if self.iteration < self.opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if self.stage == "coarse":
                    densify_threshold = self.opt.densify_grad_threshold_coarse
                else:
                    densify_threshold = self.opt.densify_grad_threshold_fine_init - self.iteration*(self.opt.densify_grad_threshold_fine_init - self.opt.densify_grad_threshold_after)/(self.opt.densify_until_iter)

                if  self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0 and self.gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify(densify_threshold,self.scene.cameras_extent)
                
                if  self.iteration > self.opt.pruning_from_iter and self.iteration % self.opt.pruning_interval == 0 and self.gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.prune(self.hyperparams.opacity_lambda, self.scene.cameras_extent, size_threshold, self.iteration % self.opt.opacity_reset_interval == 0)

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
            
            if (self.iteration in self.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
                torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + f"_{self.stage}_" + str(self.iteration) + ".pth")

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


        if not os.path.exists(f'debugging/{self.iteration}') and self.iteration % 1000 == 0:
            os.mkdir(f'debugging/{self.iteration}')

        PSNR = 0.
        SSIM = 0.

        idx = 0
        # Render and return preds
        for viewpoint_cam in tqdm(viewpoint_cams, desc="Processing viewpoints"):
            try: # If we have seperate depth
                viewpoint_cam, depth_cam = viewpoint_cam
            except:
                depth_cam = None

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, stage='fine', cam_type=self.scene.dataset_type)
            image = render_pkg["render"]


            if self.scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()

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

def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")

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
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    gui = GUI(args=args, hyperparams=hp.extract(args), dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations,
            ckpt_it=args.checkpoint_iterations, ckpt_start=args.start_checkpoint, debug_from=args.debug_from, expname=args.expname)

    
    gui.render()
    # training( args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")