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

import torch
from torch import nn
import numpy as np
from torch.onnx.symbolic_opset9 import unsqueeze

from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import math
def getProjectionMatrix_(znear, zfar, fovX, fovY, ppx, ppy, image_width, image_height):
    """
    Generate a perspective projection matrix incorporating the principal point (ppx, ppy).

    Parameters:
        znear (float): Distance to the near clipping plane.
        zfar (float): Distance to the far clipping plane.
        fovX (float): Horizontal field of view in radians.
        fovY (float): Vertical field of view in radians.
        ppx (float): Principal point x-coordinate (image center x).
        ppy (float): Principal point y-coordinate (image center y).
        image_width (float): Width of the image.
        image_height (float): Height of the image.

    Returns:
        torch.Tensor: A 4x4 perspective projection matrix.
    """
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # Adjust the projection based on the principal point
    offset_x = 2 * (ppx / image_width) - 1  # Normalized offset in [-1, 1]
    offset_y = 2 * (ppy / image_height) - 1  # Normalized offset in [-1, 1]

    # Initialize the projection matrix
    P = torch.zeros(4, 4)

    z_sign = 1.0  # Use 1.0 for a right-handed coordinate system

    # Scale factors for x and y
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)

    # Translation offsets
    P[0, 2] = (right + left) / (right - left) - offset_x
    P[1, 2] = (top + bottom) / (top - bottom) - offset_y

    # Depth scaling and perspective division
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def getProjectionMatrixFromIntrinsics(fx, fy, cx, cy, width, height, znear, zfar):
    P = torch.zeros(4, 4)

    # Map intrinsics to normalized device coordinates
    P[0, 0] = 2 * fx / width
    P[0, 2] = 1 - 2 * cx / width
    P[1, 1] = 2 * fy / height
    P[1, 2] = 2 * cy / height - 1

    # Depth (Z)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = (-znear * zfar) / (zfar - znear)
    P[3, 2] = 1.0

    return P

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, depth:bool=False,
                 weights=None,
                 cxfx=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.gt_alpha_mask = None
        if image is not None:
            if image.shape[0] == 4:
                self.gt_alpha_mask = image[-1,:]
                image = image[:-1, :]

        self.original_image = image

        try:
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        except:
            self.image_width = self.original_image.shape[1]
            self.image_height = self.original_image.shape[0]

        self.depth = depth
        self.mask = mask
        self.weights = weights
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        if cxfx is None:
            self.cx =  self.image_width / 2.0
            self.cy = self.image_height / 2.0
            fx = 0.5 * self.image_width / np.tan(self.FoVx * 0.5)
            fy = 0.5 * self.image_height / np.tan(self.FoVy * 0.5)
        else:
            self.cx = cxfx[0]
            self.cy = cxfx[1]
            fx = cxfx[2]
            fy = cxfx[3]
            
        self.focal_x = fx
        self.focal_y = fy
        # Construct intrinsics matrix
        self.intrinsics = torch.tensor([
            [fx, 0, self.cx],
            [0, fy, self.cy],
            [0,  0,      1]
        ], dtype=torch.float32)


        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)

        # self.projection_matrix = getProjectionMatrixFromIntrinsics(fx, fy, self.cx, self.cy, self.image_width, self.image_height, self.znear, self.zfar).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                      fovY=self.FoVy).transpose(0, 1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @torch.no_grad()
    def direction_normal(self): 
        R_c2w = self.world_view_transform[:3, :3].T
        # Camera looks along [0, 0, -1] in its local space
        direction = torch.tensor([0.0, 0.0, -1.0], device=R_c2w.device)
        direction = R_c2w @ direction  # transform to world space
        direction = direction / direction.norm()
        return direction

    def update_projections(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                      fovY=self.FoVy).transpose(0, 1)

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

