import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import json
import torch
from tqdm import tqdm
import cv2
import copy
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov

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


def decompose_dataset(datadir, rotation_correction, split='test', visualise_poses=False):
    with open(os.path.join(datadir, "calibration_noborder.json")) as f:
        calib = json.load(f)

    # Get the camera names for the current folder
    cam_names = os.listdir(os.path.join(datadir, f"{split}/"))

    with open(os.path.join(datadir, "capture-area.json")) as f:
        cap_area_config = json.load(f)


    poses = {}
    for ii, c in enumerate(cam_names):

        meta = calib[c]

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
        focal_depth = [meta['depth_intrinsics']['fx'], meta['depth_intrinsics']['fy']]

        K = np.array([[focal[0], 0, meta['colour_intrinsics']['ppx']], [0, focal[1], meta['colour_intrinsics']['ppy']], [0, 0, 1]])

        poses[c] = {
            'H': H, 'W': W,
            'focal': focal,
            'FovX': focal2fov(focal[0], W),
            'FovY': focal2fov(focal[1], H),
            'R': R,
            'T': T,
            'cx': meta['colour_intrinsics']['ppx'],
            'cy': meta['colour_intrinsics']['ppy'],
        }


    return poses, H, W


class CondenseData(Dataset):
    def __init__(
            self,
            datadir,
            split='train',
            downsample=1.0
    ):
        self.downsample = downsample

        if split == 'train':
            self.image_type_folder = "color_no_border"
        elif split == 'test':
            self.image_type_folder = "scene_masks"

        with open(os.path.join(datadir, f"rotation_correction.json")) as f:
            self.rotation_correction = json.load(f)

        self.cam_infos, self.h, self.w = decompose_dataset(datadir, self.rotation_correction, split=split)  # , visualise_poses=True)

        self.new_w, self.new_h = int(self.w/ self.downsample), int(self.h/self.downsample)

        self.transform = T.ToTensor()

        self.data, self.nerfpp_data = self.load_data(datadir, split)

    def load_image(self, directory):
        img = cv2.imread(directory, cv2.IMREAD_UNCHANGED)

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.downsample != 1.0:
            img = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LANCZOS4)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if img.shape[0] == 4:
            return img[:3, ...], img[3, ...]
        else:

            return img[:3, ...], None

    def load_data(self, cam_folder, split):
        cameras = [] # initialise the list of camera objects

        cam_centers = [] # for nerfpp normalization

        uid = 0 # initialised the unique id

        # For each camera
        for cam_info in tqdm(self.cam_infos, desc="Processing cameras"):
            meta = self.cam_infos[cam_info]

            # basic cam params
            R = meta['R']
            T = meta['T']
            fovx = meta['FovX']
            fovy = meta['FovY']

            # Computations for NeRFpp normalization
            W2C = getWorld2View2(R, T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

            # Get list of frames per camera
            fp = os.path.join(cam_folder, f"{split}/{cam_info}/{self.image_type_folder}/")
            sorted_list = sorted(os.listdir(fp), key=lambda img_fp: float(int(img_fp.split('.')[0]) / 10_000_000))
            for img_fp in tqdm(sorted_list, desc=f"Processing images for {cam_info}", leave=False):
                fid = float(int(img_fp.split('.')[0]) / 10_000_000) # calculate the time for the frame between 0 and 1

                cameras.append(
                    {
                        "uid":uid,
                        "fp":os.path.join(fp, img_fp),
                        "R":R, "T":T,
                        "fovx":fovx,"fovy":fovy,
                        "fid":fid
                    })
                uid += 1

        nerfnorm = self.NeRFppNormalization(cam_centers)

        return cameras, nerfnorm

    def NeRFppNormalization(self, cam_centers):
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal
        translate = -center

        return {"translate": translate, "radius": radius}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]

        gt_image, mask = self.load_image(img['fp'])
        return Camera(colmap_id=img['uid'], R=img['R'], T=img['T'],
               FoVx=img['fovx'], FoVy=img['fovy'],
               image=gt_image, gt_alpha_mask=mask,
               image_name=img['fp'], uid=img['uid'],
               data_device='cuda', fid=img['fid'],
               depth=None, flow_dirs=[])

    def copy(self):
        return copy.deepcopy(self)

    def pop(self, index):
        img = self.data.pop(index)

        gt_image, mask = self.load_image(img['fp'])

        return Camera(colmap_id=img['uid'], R=img['R'], T=img['T'],
                      FoVx=img['fovx'], FoVy=img['fovy'],
                      image=gt_image, gt_alpha_mask=mask,
                      image_name=img['fp'], uid=img['uid'],
                      data_device='cuda', fid=img['fid'],
                      depth=None, flow_dirs=[])




