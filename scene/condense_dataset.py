import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from torchvision import transforms as T
import torch.nn.functional as F
import json
import torch

import cv2

def to_tensor(x, shape=None, device="cpu"):
    # convert the input to torch.tensor
    if shape != None:
        return torch.tensor(x, dtype=torch.float32).view(shape).to(device)
    else:
        return torch.tensor(x, dtype=torch.float32).to(device)


def default_deform_tracking(config, device):
    """ to apply the inverse of [R|T] to the mesh """
    R = to_tensor(config['orientation'], (3, 3), device)  # transpose is omitted to make it column-major
    invT = R @ -to_tensor(config['origin'], (3, 1), device)
    space = to_tensor(config['spacing'], (3,), device)
    dimen = to_tensor(config['dimensions'], (3,), device)

    # offset initialized to zeros
    offset = torch.zeros(invT.size()).to(device)
    offset[1] -= space[1] * (dimen[1] / 2.0)
    offset[2] -= space[2] * (dimen[2] / 2.0)

    T = invT + offset
    return R.unsqueeze(0), T.unsqueeze(0)



def generate_undistortion_grid(K, coeffs, w, h, device='cpu'):
    """
    Generate a grid for undistorting an image with 8 distortion coefficients.

    Parameters:
        K (torch.Tensor): Camera matrix (3x3).
        coeffs (torch.Tensor): Distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6).
        w (int): Image width.
        h (int): Image height.
        device (str): Device for the tensors ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Undistortion grid of shape (1, h, w, 2).
    """
    # Generate pixel grid
    i = torch.linspace(0, h - 1, h, device=device)
    j = torch.linspace(0, w - 1, w, device=device)
    jj, ii = torch.meshgrid(j, i, indexing='xy')

    # Normalize coordinates to [-1, 1]
    x = (jj - K[0, 2]) / K[0, 0]
    y = (ii - K[1, 2]) / K[1, 1]

    # Calculate radial distance squared
    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3

    # Radial distortion
    radial = 1 + coeffs[0] * r2 + coeffs[1] * r4 + coeffs[4] * r6 + coeffs[5] * r4 + coeffs[6] * r6

    # Tangential distortion
    x_tangential = 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x ** 2)
    y_tangential = 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y ** 2)

    # Distorted coordinates
    x_distorted = x * radial + x_tangential
    y_distorted = y * radial + y_tangential

    # Convert to sampling grid format
    x_mapped = (x_distorted * K[0, 0] + K[0, 2]) / (w - 1) * 2 - 1
    y_mapped = (y_distorted * K[1, 1] + K[1, 2]) / (h - 1) * 2 - 1

    grid = torch.stack((x_mapped, y_mapped), dim=-1).unsqueeze(0)  # Shape (1, h, w, 2)
    return grid

def decompose_dataset(datadir, rotation_correction, split='test', visualise_poses=False):
    with open(os.path.join(datadir, "calibration.json")) as f:
        calib = json.load(f)

    # Get the camera names for the current folder
    cam_names = os.listdir(os.path.join(datadir, f"{split}/"))

    with open(os.path.join(datadir, "capture-area.json")) as f:
        cap_area_config = json.load(f)


    poses = {}
    for ii, c in enumerate(cam_names):

        meta = calib['cameras'][c]

        depth_ex = meta['depth_extrinsics']
        col2depth_ex = meta['colour_to_depth_extrinsics']

        # Construct w2c transform for depth images
        M_depth = torch.eye(4)
        M_depth[:3, :3] = torch.tensor(depth_ex['orientation']).view((3, 3)).mT
        M_depth[:3, 3] = torch.tensor(depth_ex['translation']).view((3, 1))[:, 0]

        # Construct w2c transform for depth images
        M_col = torch.eye(4)
        M_col[:3, :3] = torch.tensor(col2depth_ex['orientation']).view((3, 3)).mT
        M_col[:3, 3] = torch.tensor(col2depth_ex['translation']).view((3, 1))[:, 0]

        R_m, T_m = default_deform_tracking(cap_area_config, 'cpu')
        M_m = torch.eye(4)
        M_m[:3, :3] = torch.tensor(R_m).view((3, 3))
        M_m[:3, 3] = torch.tensor(T_m).view((3, 1))[:, 0]


        # Generate color (c2w transform) extrinsics for
        M = M_col.inverse() @ M_depth.inverse() @ M_m.inverse()
        M = M #.inverse()
        T = M[:3, 3].numpy()
        R = M[:3, :3].numpy()
        R = R.T

        M_d = M_depth.inverse() @ M_m.inverse()
        M_d = M_d #.inverse()
        T_d = M_d[:3, 3].numpy()
        R_d = M_d[:3, :3].numpy()
        R_d = R_d.T

        H = meta['colour_intrinsics']['height']
        W = meta['colour_intrinsics']['width']
        focal = [meta['colour_intrinsics']['fx'], meta['colour_intrinsics']['fy']]

        # K = np.array([[focal[0], 0, meta['colour_intrinsics']['ppx']], [0, focal[1], meta['colour_intrinsics']['ppy']], [0, 0, 1]])

        poses[c] = {
            'H': H, 'W': W,
            'focal': focal,
            'FovX': focal2fov(focal[0], W),
            'FovY': focal2fov(focal[1], H),
            'R': R,
            'T': T,
            'cx': meta['colour_intrinsics']['ppx'],
            'cy': meta['colour_intrinsics']['ppy'],
            
            'depth':{
                'H': meta['depth_intrinsics']['height'], 'W': meta['depth_intrinsics']['width'],
                'focal': (meta['depth_intrinsics']['fx'], meta['depth_intrinsics']['fy']),
                'FovX': focal2fov(meta['depth_intrinsics']['fx'], meta['depth_intrinsics']['width']),
                'FovY': focal2fov(meta['depth_intrinsics']['fy'], meta['depth_intrinsics']['height']),
                'R': R_d,
                'T': T_d,
                'cx': meta['depth_intrinsics']['ppx'],
                'cy': meta['depth_intrinsics']['ppy'],
            }
        }


    return poses, H, W


class CondenseData(Dataset):
    def __init__(
            self,
            datadir,
            split='train',
            downsample=1.0
    ):

        if split == 'train':
            self.image_type_folder = "color_corrected"
        elif split == 'test':
            self.image_type_folder = "scene_masks"
            
        with open(os.path.join(datadir, f"rotation_correction.json")) as f:
            self.rotation_correction = json.load(f)
             
            
        self.root_dir = datadir        
        self.downsample = downsample

        self.split = split
        self.num_frames = 0

        self.cam_infos, self.H, self.W = decompose_dataset(datadir, self.rotation_correction, split=split ) #, visualise_poses=True)
        self.new_w, self.new_h = int(self.W/ self.downsample), int(self.H/self.downsample)
        
        self.transform = T.ToTensor()

        self.image_paths, self.image_poses, self.image_times, self.fovs,self.image_centers = self.load_images_path(self.root_dir, self.split)
        self.pcd_paths = self.load_pcd_path()

        # Finally figure out idx of coarse image
        self.stage = 'coarse'
        self.get_mask = False
    
    def load_image(self, directory):
        
        img = cv2.imread(directory, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {directory}")
        
        if self.split == 'train':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.split == 'test':
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        
        if self.downsample != 1.0:
            img = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.0 
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]

        return img

    def depth_image(self, directory):
        
        img = cv2.imread(directory, cv2.IMREAD_UNCHANGED)
        img = torch.from_numpy(img.astype(np.float32)) # [C, H, W]

        return img
        
    def load_images_path(self, cam_folder, split,  stage='fine'):
        image_paths = []
        image_poses = []
        image_times = []
        image_centers = []
        
        

        FOVs = []
        self.poses = []
        
        self.mask_idxs = []
        static_mask_fps = os.listdir(os.path.join(self.root_dir, 'static_masks'))
        
        idxs = 0
        for cam_info in self.cam_infos:
            meta = self.cam_infos[cam_info]
            
            if f'{cam_info}.png' in static_mask_fps:
                self.mask_idxs.append(idxs)

            self.poses.append((meta['R'],meta['T']))
            self.focal = meta['focal']

            fovx = meta['FovX']
            fovy = meta['FovY']

            fp = os.path.join(cam_folder, f"{split}/{cam_info}/{self.image_type_folder}")
            list_dir = sorted(os.listdir(fp), key=lambda x: int(x.split('.')[0]))
            time_max = len(list_dir)
            cnt = 0
            for img_fp in list_dir:
                img_fp_ = os.path.join(fp, img_fp)
                image_paths.append(img_fp_)
                image_poses.append((meta['R'], meta['T']))
                image_times.append(float(int(img_fp.split('.')[0]) / 10_000_000))

                FOVs.append((fovx, fovy))
                image_centers.append((meta['cx'], meta['cy'], meta['focal'][0], meta['focal'][1]))
    
                cnt+= 1
                idxs += 1
        
        depth_paths = []
        depth_poses = []
        depth_times = []
        depth_FOVs = []
        for cam_info in self.cam_infos:
            meta = self.cam_infos[cam_info]

            fovx = meta['FovX']
            fovy = meta['FovY']

            fp = os.path.join(cam_folder, f"{split}/{cam_info}/{self.image_type_folder.replace('color_corrected', 'depth_corrected')}")
            list_dir = sorted(os.listdir(fp), key=lambda x: int(x.split('.')[0]))
            cnt = 0
            for img_fp in list_dir:
                img_fp_ = os.path.join(fp, img_fp)
                depth_paths.append(img_fp_)
                depth_poses.append((meta['R'], meta['T']))
                depth_times.append(float(int(img_fp.split('.')[0]) / 10_000_000))

                depth_FOVs.append((fovx, fovy))
                
        self.num_frames = cnt
        
        self.depth_data = {
            "paths":depth_paths,
            "poses":depth_poses,
            "times":depth_times,
            "fovs":depth_FOVs
        }
        return image_paths, image_poses, image_times, FOVs, image_centers

    def load_pcd_path(self):
        fp = os.path.join(self.root_dir, f"pcds/sparse/")

        fps = []
        for f in os.listdir(fp):
            fps.append(os.path.join(fp, f))

        return fps

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        mask = None
        depth = None
        path = self.image_paths[index]
        pose = self.image_poses[index]
        time = self.image_times[index]
        centers = self.image_centers[index] 
        if self.split == 'train':
            if self.get_mask:
                camid = os.path.join(f'{self.root_dir}/static_masks', path.split('/')[-3])
                camid = f'{camid}.png'
                mask = self.load_image(camid)
                mask = (mask.sum(0) > 0).float()
        img = self.load_image(path)
        depth = self.load_image(path.replace('color_corrected', 'depth_mono'))
        return img, pose, time, mask, depth, centers
    
    def get_depth(self, index):
        path = self.depth_data['paths'][index]
        pose = self.depth_data['poses'][index]
        time = self.depth_data['times'][index]
        fov = self.depth_data['fovs'][index]

        img = self.depth_image(path)
        return img, pose, time, fov

    def get_pcd_path(self, index):
        try:
            return self.pcd_paths[index % self.num_frames]
        except:
            return ''

    def load_pose(self, index):
        return self.image_poses[index]

    def load_fov(self, index):
        return self.fovs[index]


import matplotlib.pyplot as plt

def show_rgb_image(tensor):
    """
    Displays an RGB image tensor of shape [3, H, W].
    """
    img = tensor.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] and to numpy
    plt.imshow(img)
    plt.axis('off')
    plt.show()