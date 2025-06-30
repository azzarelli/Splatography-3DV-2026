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

import os
import random
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, num_cams='4', load_iteration=None, skip_coarse=None, max_frames=50):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(os.path.join(args.source_path, "rotation_correction.json")):
            scene_info = sceneLoadTypeCallbacks["Condense"](args.source_path, args.resolution)
            dataset_type="condense"
            max_frames = 300
            num_cams = 4
        else:
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, '4', max_frames)
            dataset_type="dynerf"
            max_frames = 50
            num_cams = 4

        self.maxtime = scene_info.maxtime
        self.maxframes = max_frames
        self.num_cams = num_cams
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type, 'train', maxframes=max_frames, num_cams=num_cams)
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type, 'test', maxframes=max_frames, num_cams=num_cams)

        if skip_coarse:
            print(f'Skipping coarse step with {skip_coarse}')
            self.gaussians.load_ply(os.path.join(skip_coarse,"point_cloud.ply"))
            self.gaussians.load_model(skip_coarse)
        elif self.loaded_iter:
            print(f'Load from iter {self.loaded_iter}')

            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                ))
        else:
            print('Creating init from point cloud ...')
            self.getTrainCameras().dataset.get_mask = True
            zero_cams = [self.getTrainCameras()[idx] for idx in self.train_camera.zero_idxs]
            self.getTrainCameras().dataset.get_mask = False

            self.gaussians.create_from_pcd(scene_info.point_cloud, zero_cams, dataset_type)

        if not skip_coarse and load_iteration is None:
            with torch.no_grad():
                self.train_camera.update_target(self.gaussians._xyz[~self.gaussians.target_mask].mean(dim=0).cpu())
        
        if self.dataset_type == "condense":
            from scene.dataset_readers import format_condense_infos
            self.video_camera  = format_condense_infos(scene_info.train_cameras, "val", pos=self.gaussians.get_xyz[self.gaussians.target_mask].mean(1))

        
    def get_pseudo_view(self):
        """Generate a pseudo view with four known cameras 
        """
        return self.train_camera.get_novel_view_from_config()

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def init_fine(self):
        self.train_camera.dataset.stage = 'fine'
        self.test_camera.dataset.stage = 'fine'

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTrainCamerasZero(self, scale=1.0):
        return self.train_camera
    
    def index_train(self, index):
        return self.train_camera[index]
    
    def getTestCameras(self, scale=1.0):
        return self.test_camera
    
    def getVideoCameras(self, scale=1.0):
        return self.video_camera