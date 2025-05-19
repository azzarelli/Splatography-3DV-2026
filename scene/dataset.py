from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
import cv2
from scipy.spatial.transform import Rotation as scipy_R
from scipy.spatial.transform import Slerp

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        split,
        maxframes
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        
        self.zero_idxs = [i*maxframes for i in range(4)]
        if dataset_type == 'train':
            print(f'Zero Indexs are: {self.zero_idxs}')
            
        self.vert_poses = self.dataset.poses

        self.fovx = focal2fov(self.dataset.focal[0], self.dataset.W)
        self.fovy = focal2fov(self.dataset.focal[1], self.dataset.H)
        
        if split == 'train':
            T_list = [torch.from_numpy(t[1]).cpu().float() for t in self.vert_poses]
            self.T_stack = torch.stack(T_list) 
            self.AA = self.T_stack.min(dim=0).values
            self.BB = self.T_stack.max(dim=0).values
            
            
            rotations = [t[0] for t in self.vert_poses]
            self.rots = scipy_R.from_matrix(rotations)
            self.rotations = [t[0] for t in self.vert_poses]

    
    def update_target(self, mean):
        self.mean = mean
        
        # self.mean = triangulate_from_rays(self.T_stack, torch.cat(self.rotations, dim=0)).cpu()

    
    def get_novel_view_from_config(self):
        """Return a novel view camera based off the initial training set (R,T)
        """
        T_nv = self.AA + (self.BB - self.AA) * torch.rand(3).cpu()
        
        R =  look_at_rotation(T_nv, self.mean).numpy()

        T = T_nv.numpy()
        
        random_cam = Camera(colmap_id=0, R=R, T=T, FoVx=self.fovx, FoVy=self.fovy, image=None, gt_alpha_mask=None,
            image_name=f"pseudo", uid=0, data_device=torch.device("cuda"), time=0., mask=None
        )
        random_cam.image_height = int(self.dataset.H)
        random_cam.image_width = int(self.dataset.W)

        return random_cam

    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time, mask, depth, weights = self.dataset[index]
                R, T = w2c
                FovX = self.fovx
                FovY = self.fovy
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time

                mask = caminfo.mask
            
            return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time, 
                          mask=mask, 
                          depth=depth,
                          weights=weights
                      )
        else:
            return self.dataset[index]


    def __len__(self):
        
        return len(self.dataset)


def look_at_rotation(camera_position: torch.Tensor, target_position: torch.Tensor):
    up: torch.Tensor = torch.tensor([0.0, 1.0, 0.0]).float()
    """
    Construct a 3x3 rotation matrix that orients the camera to look at a target.
    """
    # Forward vector (camera's z axis): pointing from camera to target
    forward = target_position - camera_position 
    forward = forward / forward.norm()

    right = torch.cross(up, forward)
    right = right / right.norm()

    true_up = torch.cross(forward, right)
    true_up = true_up / true_up.norm()

    R = torch.stack([right, true_up, forward], dim=1)  # shape: (3, 3)
    return R.T

def triangulate_from_rays(camera_positions: torch.Tensor, rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Estimate 3D target position from camera positions and rotation matrices.
    Assumes the camera looks along its +Z axis (i.e., forward = R[:, 2]).
    
    Inputs:
        camera_positions: (N, 3)
        rotation_matrices: (N, 3, 3)
    
    Returns:
        target_position: (3,)
    """
    N = camera_positions.shape[0]

    # Get the viewing directions (z-axis of each rotation matrix)
    directions = rotation_matrices[:, :, 2]  # shape: (N, 3)

    # Normalize the directions
    directions = directions / directions.norm(dim=1, keepdim=True)

    # Set up least squares problem: A x = b
    A = torch.zeros((3, 3)).float()
    b = torch.zeros(3).float()
    I = torch.eye(3).float()

    for i in range(N):
        d = directions[i].float()
        o = camera_positions[i]
        A_i = I - d[:, None] @ d[None, :]  # Projection matrix
        A += A_i

        b += A_i @ o

    # Solve for x: A x = b
    target_position = torch.linalg.solve(A, b)
    return target_position