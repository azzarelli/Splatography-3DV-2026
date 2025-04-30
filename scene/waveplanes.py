from typing import Optional

from submodules.pytorch_wavelets_.dwt.transform2d import DWTInverse

from utils.kplane_utils import grid_sample_wrapper, normalize_aabb, GridSet

import torch
import torch.nn as nn
import torch.nn.functional as F

OFFSETS = torch.tensor([
    [-1.0, 0.0],
    [-0.5, 0.0],
    [0.5, 0.0],
    [1.0, 0.0],
    [0.0, -1.0],
    [0.0, -0.5],
    [0.0, 0.5],
    [0.0, 1.0],
    [0.5, 0.5],
    [0.5, -0.5],
    [-0.5, 0.5],
    [-0.5, -0.5]
]).cuda().unsqueeze(0)

def interpolate_features_MUL(pts: torch.Tensor,time, kplanes, idwt, is_opacity_grid, scales):
    """Generate features for each point
    """
    # time m feature
    interp_1 = 1.

    # samples = rays_pts_emb[mask,:3].unsqueeze(1).repeat(1,13,1)
    # base_offsets = torch.tensor([-1.0, -0.5, 0.5, 1.0], device=mask.device).unsqueeze(0).repeat(3,1)  # (4,)
    
    # scales = scale_emb[mask, :3].unsqueeze(1).repeat(1,4,1)
    
    # # scale_offsets =  
    
    # print(samples.shape, base_offsets.shape, scales.shape)
    # exit()
    
    # q,r are the coordinate combinations needed to retrieve pts
    q, r = 0, 1
    for i in range(3):
        scale_offsets = scales[..., (q,r)].unsqueeze(1).repeat(1,12,1) * OFFSETS

        pts_ = pts[..., (q, r)].unsqueeze(1).repeat(1,13,1)
        pts_[:, 1:, :] += scale_offsets

        pts_ = pts_.view(-1, 2)
        
        feature = kplanes[i](pts_, idwt)
        
        feature = feature.view(-1,13,feature.shape[-1]).mean(1)

        interp_1 = interp_1 * feature

        r +=1
        if r == 3:
            q = 1
            r = 2
    return interp_1
   


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
        
        for i in range(3):
            what = 'space'
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
        for i in range(3):
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

        
    def get_opacity_vars(self, pts):
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, pts.shape[-1])
        return interpolate_features_MUL(
            pts,None, self.grids, self.idwt, True
        )
        
    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None, scales=None):
        pts = normalize_aabb(pts, self.aabb)
        
        timestamps = timestamps.detach().clone()
        # timestamps = (timestamps*2.)-1. # normalize timestamps between 0 and 1

        pts = pts.reshape(-1, pts.shape[-1])

        return interpolate_features_MUL(
            pts,timestamps, self.grids, self.idwt, False,scales)
