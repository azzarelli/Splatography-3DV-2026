import concurrent.futures
import gc
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
import torch.nn.functional as F

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


class Neural3D_NDC_Dataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=True,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        bd_factor=0.75,
        eval_step=1,
        selected_cams=[],
        maxframes=50,
        sphere_scale=1.0,
    ):
        
        self.img_wh = (
            int(2704 / downsample),
            int(2028 / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.depth_paths = []
        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.selected_cams = selected_cams
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False
        self.maxframes = maxframes
        
        self.get_mask = False
        self.get_depth = False

        self.stage = 'rgb'
        self.load_meta()
        print(f"meta data loaded, total image:{len(self)}")


    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]
        videos = glob.glob(os.path.join(self.root_dir, "cam*.mp4"))
        videos = sorted(videos)
        # breakpoint()
        assert len(videos) == poses_arr.shape[0]

        selected_idxs = []
        for idx, vid in enumerate(videos):
            for s in self.selected_cams:
                if f'{s:02}' in vid.split('/')[-1]:
                    selected_idxs.append(idx)
        self.selected_cams = selected_idxs

        self.H, self.W, focal = poses[0, :, -1] / self.downsample
        
        self.focal = [focal, focal]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

        # Sample N_views poses for validation - NeRF-like camera trajectory.
        self.val_poses = get_spiral(poses, self.near_fars, N_views=self.maxframes)
        # self.val_poses = self.directions
        self.poses = poses[self.selected_cams]
        self.poses_all = poses #[[0]+poses_i_train]
        self.image_paths, self.image_poses, self.image_times, N_cam, N_time, self.poses = self.load_images_path(videos, self.split)
        self.cam_number = N_cam
        self.time_number = N_time
        
    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times
    
    def load_images_path(self,videos,split):
        image_paths = []
        image_poses = []
        image_times = []
        
        general_poses = []
        
        N_cams = 0
        N_time = 0
        for index, video_path in enumerate(videos):

            if (index in  self.selected_cams and split == 'train') or (index == 0 and split == 'test'):
                N_cams +=1
                count = 0
                video_images_path = video_path.split('.')[0]
                if split == 'train':
                    if 'flame_steak' in video_images_path: name__ = "images_2x"
                    else: name__ = 'images'
                    image_path = os.path.join(video_images_path,name__)
                    depth_path = video_images_path[:-5]+'vda/'+video_images_path.split('/')[-1]+'/target_depth.png'
                    self.depth_paths.append(depth_path)
                else:
                    image_path = os.path.join(video_images_path,"masks")

                video_frames = cv2.VideoCapture(video_path)
                if not os.path.exists(image_path):
                    print(f"no images saved in {image_path}, extract images from video.")
                    os.makedirs(image_path)
                    this_count = 0
                    while video_frames.isOpened():
                        ret, video_frame = video_frames.read()
                        if this_count >= self.maxframes:break
                        if ret:
                            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                            video_frame = Image.fromarray(video_frame)
                            if self.downsample != 1.0:

                                img = video_frame.resize(self.img_wh, Image.LANCZOS)
                            img.save(os.path.join(image_path,"%04d.png"%count))

                            count += 1
                            this_count+=1
                        else:
                            break
                        
                images_path = os.listdir(image_path)
                images_path.sort()
                this_count = 0
                for idx, path in enumerate(images_path):
                    if this_count >=self.maxframes:break
                    image_paths.append(os.path.join(image_path,path))
                    pose = np.array(self.poses_all[index])
                    R = pose[:3,:3]
                    R = -R
                    R[:,0] = -R[:,0]
                    T = -pose[:3,3].dot(R)
                    image_times.append(idx/(self.maxframes-1))
                    
                    if idx == 0:
                        general_poses.append((R,T))
                    image_poses.append((R,T))
                    this_count+=1
                # print(video_images_path, R,T)
                N_time = len(images_path)
        # exit()
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # positions = np.array([T for R, T in image_poses])
        # x = positions[:, 0]
        # y = positions[:, 1]
        # z = positions[:, 2]

        # ax.scatter(x, y, z, marker='o')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('3D Scatter of Positions')
        # plt.show()
        # exit()
        return image_paths, image_poses, image_times, N_cams, N_time, general_poses
    def __len__(self):
        return len(self.image_paths)
    
    def get_depth_img(self, index):
        img = Image.open(self.depth_paths[index])
        img = self.transform(img)
        return img
    
    def __getitem__(self,index):
        img = Image.open(self.image_paths[index])
        
        img = self.transform(img)
        extra = None
        if self.get_mask:
            if 'cam00' not in self.image_paths[index] and'0000.png' in self.image_paths[index]:
                camid = os.path.join(f'{self.root_dir}/static_masks',self.image_paths[index].split('/')[-3])
                camid = f'{camid}.png'
                extra = Image.open(camid)
                extra = extra.resize((img.shape[-1], img.shape[-2]), Image.LANCZOS)

                extra = self.transform(extra)[-1]
        # elif self.get_depth:
        else:
            if 'cam00' not in self.image_paths[index]:
                extra = Image.open(self.image_paths[index].replace('images', 'depth'))
                extra = self.transform(extra).float()/255.

        return img, self.image_poses[index], self.image_times[index], extra
    
    
    def load_pose(self,index):
        return self.image_poses[index]

