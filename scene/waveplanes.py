from typing import Optional

from submodules.pytorch_wavelets_.dwt.transform2d import DWTInverse

from utils.kplane_utils import grid_sample_wrapper, normalize_aabb, GridSet

import torch
import torch.nn as nn
import torch.nn.functional as F

# OFFSETS = torch.tensor([
#     [-1.0, 0.0],
#     [-0.5, 0.0],
#     [0.5, 0.0],
#     [1.0, 0.0],
#     [0.0, -1.0],
#     [0.0, -0.5],
#     [0.0, 0.5],
#     [0.0, 1.0],
#     [0.5, 0.5],
#     [0.5, -0.5],
#     [-0.5, 0.5],
#     [-0.5, -0.5]
# ]).cuda().unsqueeze(0)

def interpolate_features_MUL(pts, time, kplanes, idwt, scales):
    """Generate features for each point
    """
    # time m feature
    space = 1.
    spacetime = 1.
    
    data = torch.cat([pts, time], dim=-1)
    
    # q,r are the coordinate combinations needed to retrieve pts
    coords = [[0,1], [0,2],[0,3], [1,2], [1,3], [2,3]]
    for i in range(len(coords)):
        q,r = coords[i]

        if i in [0,1,3]:
            # scale_offsets = scales[..., (q,r)].unsqueeze(1).repeat(1,12,1) * OFFSETS

            # data_ = data[..., (q, r)].unsqueeze(1).repeat(1,13,1)
            # data_[:, 1:, :] += scale_offsets

            # data_ = data_.view(-1, 2)

            # feature = kplanes[i](data_, idwt)
            
            # feature = feature.view(-1,13,feature.shape[-1]).mean(1)
            feature = kplanes[i](data[..., (q,r)], idwt)
            space = space * feature

        elif i in [2, 4, 5]:
            spacetime = spacetime * kplanes[i](data[..., (r, q)], idwt)

    return space, spacetime
   

def interpolate_features_theta(pts, angle, kplanes, idwt):
    """Generate features for each point
    """
    feature = 1.
    
    data = torch.cat([pts, angle], dim=-1)
    
    # q,r are the coordinate combinations needed to retrieve pts
    coords = [[0,3], [1,3],[2,3]]
    for i in range(len(coords)):
        q,r = coords[i]

        feature = feature * kplanes[6+i](data[..., (q,r)], idwt)
    return feature
   

import matplotlib.pyplot as plt

def visualize_grid_and_coords(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True):
    """
    Visualizes the mean grid (averaged over batch) and overlays the sampling coordinates.
    
    Args:
        grid (torch.Tensor): Input tensor of shape [1, B, H, W] or [B, H, W].
        coords (torch.Tensor): Normalized coordinates in [-1, 1] of shape [N, 2] or [1, N, 2].
        align_corners (bool): Whether the coords use align_corners=True (affects projection).
    """
    # Remove singleton channel dimension if present (e.g. [1, B, H, W] -> [B, H, W])
    if grid.dim() == 4 and grid.shape[0] == 1:
        grid = grid.squeeze(0)
    
    if grid.dim() != 3:
        raise ValueError("Expected grid shape [B, H, W]")

    # Mean across batch axis
    grid_mean = grid.mean(dim=0)  # [H, W]

    H, W = grid_mean.shape
    # Handle coordinate dimensions
    if coords.dim() == 3:
        coords = coords.squeeze(0)  # [N, 2]

    if coords.shape[-1] != 2:
        raise ValueError("Coordinates must be 2D")
    
    # Convert normalized coordinates [-1, 1] to image coordinates
    def denorm_coords(coords, H, W, align_corners):
        if align_corners:
            x = ((coords[:, 0] + 1) / 2) * (W - 1)
            y = ((coords[:, 1] + 1) / 2) * (H - 1)
        else:
            x = ((coords[:, 0] + 1) * W - 1) / 2
            y = ((coords[:, 1] + 1) * H - 1) / 2
        return x, y

    x, y = denorm_coords(coords, H, W, align_corners)

    # exit()
    # Plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(grid_mean.cpu().numpy(), cmap='gray', origin='upper')
    plt.scatter(x.cpu().numpy(), y.cpu().numpy(), color='red', s=20)
    plt.title("Grid Mean with Sampled Coords")
    plt.axis('off')
    plt.show()

class WavePlaneField(nn.Module):
    def __init__(
            self,
            bounds,
            planeconfig,
            rotate=False
    ):
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.concat_features = True
        self.grid_config = planeconfig
        self.feat_dim = self.grid_config["output_coordinate_dim"]

        # 1. Init planes
        self.grids = nn.ModuleList()

        # Define the DWT functon
        self.idwt = DWTInverse(wave='coif4', mode='periodization').cuda().float()

        self.cacheplanes = True
        self.is_waveplanes = True
        
        for i in range(6):
            if i in [0,1,3]:
                what = 'space'
            else:
                what = 'spacetime'
            res = [self.grid_config['resolution'][0],
                self.grid_config['resolution'][1]]
            
            gridset = GridSet(
                what=what,
                resolution=res,
                J=self.grid_config['wavelevel'],
                config={
                    'feature_size': self.grid_config["output_coordinate_dim"],
                    'a': 0.1,
                    'b': 0.5,
                    'wave': 'coif4',
                    'wave_mode': 'periodization',
                },
                cachesig=self.cacheplanes
            )

            self.grids.append(gridset)

        for i in range(3):
            what = 'spacetime'
            res = [self.grid_config['resolution'][0],
                self.grid_config['resolution'][0]]
            
            gridset = GridSet(
                what=what,
                resolution=res,
                J=self.grid_config['wavelevel'],
                config={
                    'feature_size': self.grid_config["output_coordinate_dim"],
                    'a': 0.1,
                    'b': 0.5,
                    'wave': 'coif4',
                    'wave_mode': 'periodization',
                },
                cachesig=self.cacheplanes
            )
            self.grids.append(gridset)


    def compact_save(self, fp):
        import lzma
        import pickle
        data = {}

        for i in range(6):
            data[f'{i}'] = self.grids[i].compact_save()

        with lzma.open(f"{fp}.xz", "wb") as f:
            pickle.dump(data, f)

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def set_aabb(self, xyz_max, xyz_min):
        try:
            aabb = torch.tensor([
                xyz_max,
                xyz_min
            ], dtype=torch.float32)
        except:
            aabb = torch.stack([xyz_max, xyz_min], dim=0)  # Shape: (2, 3)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print("Voxel Plane: set aabb=", self.aabb)

    def grids_(self, regularise_wavelet_coeff: bool = False, time_only: bool = False, notflat: bool = False):
        """Return the grids as a list of parameters for regularisation
        """
        ms_planes = []
        for i in range(len(self.grids)):
            gridset = self.grids[i]

            if self.cacheplanes:
                ms_feature_planes = gridset.signal
            else:
                ms_feature_planes = gridset.idwt_transform(self.idwt)

            # Initialise empty ms_planes
            if ms_planes == []:
                ms_planes = [[] for j in range(len(ms_feature_planes))]

            for j, feature_plane in enumerate(ms_feature_planes):
                ms_planes[j].append(feature_plane)

        return ms_planes

    def forward(self,pts,time,scales):
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, pts.shape[-1])
        time = (time*2.)-1. # go from 0 to 1 to -1 to +1 for grid interp

        return interpolate_features_MUL(
            pts,time, self.grids, self.idwt,scales)

    def theta(self, pts, angle):
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, pts.shape[-1])
        return interpolate_features_theta(
            pts,angle, self.grids, self.idwt)