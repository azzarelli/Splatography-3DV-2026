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

def build_cov_matrix_torch(cov6):
    N = cov6.shape[0]
    cov = torch.zeros((N, 3, 3), device=cov6.device, dtype=cov6.dtype)
    cov[:, 0, 0] = cov6[:, 0]  # σ_xx
    cov[:, 0, 1] = cov[:, 1, 0] = cov6[:, 1]  # σ_xy
    cov[:, 0, 2] = cov[:, 2, 0] = cov6[:, 2]  # σ_xz
    cov[:, 1, 1] = cov6[:, 3]  # σ_yy
    cov[:, 1, 2] = cov[:, 2, 1] = cov6[:, 4]  # σ_yz
    cov[:, 2, 2] = cov6[:, 5]  # σ_zz
    return cov

def sample_from_cov(points, cov6, M):
    N = points.shape[0]
    cov = build_cov_matrix_torch(cov6)  # (N, 3, 3)
    
    # Cholesky decomposition: cov = L @ L.T
    L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(3, device=points.device))  # (N, 3, 3)
    
    # Sample standard normal noise: (N, M, 3)
    eps = torch.randn(N, M, 3, device=points.device)
    
    # Transform noise by covariance and add mean
    # L: (N, 3, 3), eps: (N, M, 3) → (N, M, 3)
    samples = torch.einsum('nij,nmj->nmi', L, eps) + points[:, None, :]
    
    return samples 

def interpolate_features_MUL(data, M, kplanes):
    """Generate features for each point
    """
    # time m feature
    space = 1.
    spacetime = 1.
    

    # q,r are the coordinate combinations needed to retrieve pts
    coords = [[0,1], [0,2],[3,0], [1,2], [3,1], [3,2]]
    for i in range(len(coords)):
        q,r = coords[i]
        feature = kplanes[i](data[..., (q, r)])
        feature = feature.view(-1, M, feature.shape[-1]).mean(dim=1)
        # feature = torch.prod(feature, dim=1)

        if i in [0,1,3]:
            space = space * feature

        elif i in [2, 4, 5]:
            spacetime = spacetime * feature

    return space, spacetime
   

def interpolate_features_theta(pts, angle, kplanes):
    """Generate features for each point
    """
    feature = 1.
    
    data = torch.cat([pts, angle], dim=-1)
    
    # q,r are the coordinate combinations needed to retrieve pts
    coords = [[0,3], [1,3],[2,3]]
    for i in range(len(coords)):
        q,r = coords[i]

        feature = feature * kplanes[6+i](data[..., (q,r)])
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
                360]
            
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

    def update_J(self):
        for grid in self.grids:
            grid.update_J()
        print(f'Updating J to {self.grids[0].current_J}')

    def grids_(self, regularise_wavelet_coeff: bool = False, time_only: bool = False, notflat: bool = False):
        """Return the grids as a list of parameters for regularisation
        """
        ms_planes = []
        for i in range(len(self.grids)):
            gridset = self.grids[i]

            ms_feature_planes = gridset.signal

            # Initialise empty ms_planes
            if ms_planes == []:
                ms_planes = [[] for j in range(len(ms_feature_planes))]

            for j, feature_plane in enumerate(ms_feature_planes):
                ms_planes[j].append(feature_plane)

        return ms_planes

    def forward(self, pts, time, cov6):
        """
            Notes:
                - to visualize samples and projection use display_projection(pts, cov6)
                    - you can modify the constants K_a and K_b to see that the samples get plotted closer to the edge
        """
        M = 13 # total of 13 samples
        pts = structured_gaussian_samples(pts, cov6)
        # display_projection(pts, cov6) # re:notes
        
        time = (time*2.)-1. # go from 0 to 1 to -1 to +1 for grid interp
        time = time.repeat(pts.shape[1], 1)
                
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, pts.shape[-1])
        
        pts = torch.cat([pts.view(-1, 3), time], dim=-1)
        
        return interpolate_features_MUL(
            pts, M, self.grids)

    def theta(self, pts, angle):
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, pts.shape[-1])
        return interpolate_features_theta(
            pts,angle, self.grids)

def display_projection(pts, cov6):
    B = 5
    mean = pts[B] 
    cov_matrix = build_cov_matrix_torch(cov6)[B]
    
    cov_xy = cov_matrix[:2, :2]
    eigvals, _ = torch.linalg.eigh(cov_xy)
    print("XY Eigenvalues:", eigvals)
    samples = samples[B]             

    display_covariance_ellipse_and_samples(mean,cov_matrix,samples)
    
    
K_a = -2*torch.log(torch.tensor(0.2)).cuda()
K_b = -2*torch.log(torch.tensor(0.8)).cuda()

def structured_gaussian_samples(pts, cov6, axis_pdf_thresholds=[0.4, 0.8],):
    """ Sample gaussians along the major axis of the eigen of the covariances
        w.r.t the Gaussian function G=exp(-0.5(x-u)^T E^-1 (x-u))
    """
    N = pts.shape[0]
    device = pts.device
    dtype = pts.dtype
    with torch.no_grad():
        cov3x3 = build_cov_matrix_torch(cov6)  # (N, 3, 3)
        eigenvalues, normals = torch.linalg.eigh(cov3x3) # should be symetric
        normals = normals.contiguous().view(-1,1, 3).detach() #.unsqueeze(1).repeat(1,3,1)
        # Get inverse covariance
        cov_inv = torch.linalg.inv(cov3x3).unsqueeze(1).repeat(1,3,1,1).view(-1, 3, 3)
        lam = torch.bmm(torch.bmm(normals, cov_inv), normals.transpose(1, 2)).squeeze(-1)  # (N,1,1)
        
        normals = normals.squeeze(1)
        
        # if we detach this, errors as a result of sampling will relate to point position rather than covariance
        p_a = torch.sqrt(K_a/lam) * normals
        p_b = torch.sqrt(K_b/lam) * normals
    
        o = pts.unsqueeze(1).repeat(1, 3, 1).view(-1, 3)
        x_a = (o + p_a).view(-1,3,3)
        x_b = (o + p_b).view(-1,3,3)
        x_a_neg = (o - p_a).view(-1,3,3)
        x_b_neg = (o - p_b).view(-1,3,3)
    samples = torch.cat([pts.unsqueeze(1), x_a, x_b, x_a_neg, x_b_neg], dim=1)
    return samples
    

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
def display_covariance_ellipse_and_samples(mean, cov3x3, samples, density_threshold=0.05):
    """
    mean: (3,) torch tensor
    cov3x3: (3, 3) torch tensor
    samples: (N, 3) torch tensor
    density_threshold: float, threshold value of the Gaussian PDF
    """
    mean = mean.detach().cpu().numpy()
    cov3x3 = cov3x3.detach().cpu().numpy()
    samples = samples.detach().cpu().numpy()

    # Project onto XY plane
    mean_xy = mean[:2]
    cov_xy = cov3x3[:2, :2]  # 2x2 submatrix

    # --- Compute Mahalanobis threshold (chi2 value) for PDF < density_threshold ---
    det = np.linalg.det(cov_xy)
    norm_const = 1.0 / (2 * np.pi * np.sqrt(det))
    chi2_val = -2 * np.log(density_threshold / norm_const)

    # Eigen-decomposition for ellipse orientation and axis lengths
    vals, vecs = np.linalg.eigh(cov_xy)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    width, height = 2 * np.sqrt(vals * chi2_val)  # full width/height
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # --- Plot ---
    fig, ax = plt.subplots()
    ellipse = Ellipse(xy=mean_xy, width=width, height=height, angle=angle,
                      edgecolor='red', fc='none', lw=2, label=f'PDF < {density_threshold}')

    ax.add_patch(ellipse)

    ax.plot(mean_xy[0], mean_xy[1], 'ro', label='Mean')
    ax.scatter(samples[:, 0], samples[:, 1], c='blue', s=20, label='Samples')

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("2D Covariance Ellipse (XY plane) with Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()