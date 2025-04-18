from typing import Optional

from submodules.pytorch_wavelets_.dwt.transform2d import DWTInverse

from utils.kplane_utils import grid_sample_wrapper, normalize_aabb, GridSet

import torch
import torch.nn as nn
import torch.nn.functional as F

def interpolate_features_MUL(pts: torch.Tensor, kplanes, idwt, is_opacity_grid, sep=False):
    """Generate features for each point
    """
    # time m feature
    interp_1 = 1.
    sp_interp = 1.
    if is_opacity_grid: # Sample just the space-only grids
        # q,r are the coordinate combinations needed to retrieve pts
        q, r = 0, 1
        for i in range(3):
            if i == 2:# Make sure we are using the space-only planes [0,1,3]
                i == 3
            coeff = kplanes[i]
                
            feature = coeff(pts[..., (q, r)], idwt)

            interp_1 = interp_1 * feature

            r +=1
            if r == 3:
                q = 1
                r = 2
        return interp_1
    else:
        # q,r are the coordinate combinations needed to retrieve pts
        q, r = 0, 1
        for i in range(6):
            coeff = kplanes[i]
            if r != 3:
                feature = coeff(pts[..., (q, r)], idwt)
                if sep:
                    sp_interp = sp_interp * feature
                else:
                    interp_1 = interp_1 * feature
            else:
                feature = coeff(pts[..., (3, q)], idwt)
                interp_1 = interp_1 * feature

            r += 1
            if r == 4:
                q += 1
                r = q + 1
        if sep: # return space and space time features seperately
            return interp_1, sp_interp
        else:
            return interp_1

def get_feature_probability(pts: torch.Tensor, kplanes):
    """Generate features for each point
    """
    pts = torch.concatenate([pts, torch.zeros_like(pts[:, 0].unsqueeze(-1), device=pts.device)], dim=-1)
    
    # time m feature
    interp = 1.
    # q,r are the coordinate combinations needed to retrieve pts
    q, r = 0, 1
    for i in range(6):
        if r == 3:
            vectorplane = kplanes[i].signal[0] # Get the feature plane
            # First get the mean along the temporal planes then get the mean across features
            vectorplane = vectorplane.mean(-1).unsqueeze(-1).mean(1).unsqueeze(0)
            # print(kplanes[i].signal[0].median(-1)[0].shape) 
            feature = (
                grid_sample_wrapper(vectorplane, pts[..., (3, q)])
                .view(-1, vectorplane.shape[1])
            ) #.mean(-1)
            
            interp = interp * feature

            # visualize_grid_and_coords(vectorplane.repeat(1,1,1, 50), pts[..., (3, q)])
            # exit()
            
        r += 1
        if r == 4:
            q += 1
            r = q + 1
    # exit()
    # Now determine the probabilites
    # \exp\left(-w^{2}\ \cdot\ \left(1-x\right)^{2}\right)
    # w was 49 previously
    return 1. - torch.exp(-9 * (1-interp)**2)

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
        
        j, k = 0, 1
        for i in range(6):
            # We only need a grid for the space-time features
            if k == 3: 
                what = 'spacetime'
                res = [self.grid_config['resolution'][j], self.grid_config['resolution'][3]]
            else:
                what = 'space'
                res = [self.grid_config['resolution'][j],
                    self.grid_config['resolution'][k]]
            
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
            k += 1
            if k == 4:
                j += 1
                k = j + 1
                

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
        if regularise_wavelet_coeff:
            # Retrive coefficients to regularise
            ms_planes = []
            for i in range(6):
                # Skip space planes in time only
                if time_only and i not in [2, 4, 5]:
                    continue

                gridset = self.grids[i]

                ms_feature_planes = gridset.wave_coefs(notflat=notflat)

                if ms_planes == []:
                    ms_planes = [[] for j in range(len(ms_feature_planes))]

                for j, feature_plane in enumerate(ms_feature_planes):
                    ms_planes[j].append(feature_plane)
        else:
            ms_planes = []
            for i in range(6):
                if time_only and i not in [2, 4, 5]:
                    continue

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

    def get_dynamic_probabilities(self,pts: torch.Tensor,):
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, pts.shape[-1])
        return get_feature_probability(
            pts, self.grids
        )
        
    def get_opacity_vars(self, pts):
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, pts.shape[-1])
        return interpolate_features_MUL(
            pts, self.grids, self.idwt, True
        )
        
    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None, sep:bool=False):
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is not None:
            timestamps = (timestamps*2.)-1. # normalize timestamps between 0 and 1
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])

        return interpolate_features_MUL(
            pts, self.grids, self.idwt, False, sep)
