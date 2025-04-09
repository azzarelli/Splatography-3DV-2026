import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

from pytorch_wavelets_.dwt.transform2d import DWTInverse, DWTForward

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]

    # Grid is range -1 to 1 and is dependant on the resolution
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def interpolate_features_MUL(pts: torch.Tensor, kplanes, idwt, ro_grid, is_opacity_grid):
    """Generate features for each point
    """
    # if ro_grid is not None:
    #     rot_pts = torch.matmul(pts[..., :3], ro_grid) # spatial rotation
    #     pts = torch.cat([rot_pts, pts[..., -1].unsqueeze(-1)], dim=-1) # keep time values

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
                sp_interp = sp_interp * feature
            else:
                feature = coeff(pts[..., (3, q)], idwt)
                interp_1 = interp_1 * feature

            r += 1
            if r == 4:
                q += 1
                r = q + 1

        return interp_1, sp_interp


def get_feature_probability(pts: torch.Tensor, kplanes, idwt, ro_grid, is_opacity_grid):
    """Generate features for each point
    """
    if ro_grid is not None:
        rot_pts = torch.matmul(pts[..., :3], ro_grid) # spatial rotation
        pts = torch.cat([rot_pts, pts[..., -1].unsqueeze(-1)], dim=-1) # keep time values

    pts = torch.concatenate([pts, torch.zeros_like(pts[:, 0].unsqueeze(-1), device=pts.device)], dim=-1)
    
    # time m feature
    interp = 1.
    # q,r are the coordinate combinations needed to retrieve pts
    q, r = 0, 1
    for i in range(6):
        if r == 3:
            vectorplane = kplanes[i].signal[0] # Get the feature plane
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
    return 1. - torch.exp(-400 * (1-interp)**2)

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

# Define the grid
class GridSet(nn.Module):

    def __init__(
            self,
            what: str,  # Space/Spacetime
            resolution: list,
            config: dict = {},
            is_proposal: bool = False,
            J: int = 3,
            cachesig: bool = True,
    ):
        super().__init__()

        self.what = what
        self.is_proposal = is_proposal
        self.running_compressed = False
        self.cachesig = cachesig

        init_mode = 'uniform'
        if self.what == 'spacetime':
            init_mode = 'ones'

        self.feature_size = config['feature_size']
        self.resolution = resolution
        self.wave = config['wave']
        self.mode = config['wave_mode']
        self.J = J

        # Initialise a signal to DWT into our initial Wave coefficients
        dwt = DWTForward(J=J, wave=config['wave'], mode=config['wave_mode']).cuda()
        init_plane = torch.empty(
            [1, config['feature_size'], resolution[0], resolution[1]]
        ).cuda()

        if init_mode == 'uniform':
            nn.init.uniform_(init_plane, a=config['a'], b=config['b'])
        elif init_mode == 'zeros':
            nn.init.zeros_(init_plane)
        elif init_mode == 'ones':
            nn.init.ones_(init_plane)
        else:
            raise AttributeError("init_mode not given")

        if self.what == 'spacetime':
            init_plane = init_plane - 1.
        (yl, yh) = dwt(init_plane)

        # Initialise coefficients
        grid_set = [nn.Parameter(yl.clone().detach())] + \
                   [nn.Parameter(y.clone().detach()) for y in yh]

        coef_scaler = [1., .2, .4, .6, .8]
        grids = []

        for i in range(self.J + 1):
            grids.append((1. / coef_scaler[i]) * grid_set[i])
        # Rescale so our initial coeff return initialisation
        self.grids = nn.ParameterList(grids)
        self.scaler = coef_scaler

        del yl, yh, dwt, init_plane, grid_set
        torch.cuda.empty_cache()

        self.step = 0.
        self.signal = 0.

    def compact_save(self):
        """Construct the dictionary containing non-zero coefficient values"""
        # Rescale coefficient values
        coeffs = []
        for i in range(self.J + 1):
            coeffs.append(self.grids[i])

        dictionary = {}
        data = {}

        for i_ in range(self.J + 1):
            cs = coeffs[i_].squeeze(0)
            n = 0.1

            lt = (cs < n)
            gt = (cs > -n)
            cs = ~(lt * gt) * cs  # .nonzero(as_tuple=True)
            nzids = (cs == 0.).nonzero(as_tuple=True)
            non_zero_mask = cs.nonzero()

            cs = cs.tolist()

            i = f'{i_}'
            data[i] = {}
            # Deal with father wavelets first (shape B, H, W)
            if i == '0':
                dictionary = {f'{k}.{l}.{m}': {'val': cs[k][l][m], 'l': f'{l}', 'm': f'{m}'} for (k, l, m) in
                              non_zero_mask.tolist()}

                # # Reformat
                for k_ in dictionary.keys():
                    k = k_.split('.')[0]
                    # Get branch keys
                    l = dictionary[k_]['l']
                    m = dictionary[k_]['m']
                    val = dictionary[k_]['val']

                    # Construct k branch
                    if k not in data[i].keys():
                        data[i][k] = {}
                    # Construct l branch
                    if l not in data[i][k].keys():
                        data[i][k][l] = {}

                    data[i][k][l][m] = val
            else:  # Deal with mother wavelets (shape B, F, H, W)
                dictionary = {f'{k}.{n}.{l}.{m}': {'val': cs[k][n][l][m], 'n': f'{n}', 'l': f'{l}', 'm': f'{m}'} for
                              (k, n, l, m) in non_zero_mask.tolist()}

                # Reformat
                for k_ in dictionary.keys():
                    k = k_.split('.')[0]
                    # Get branch keys
                    l = dictionary[k_]['l']
                    m = dictionary[k_]['m']
                    n = dictionary[k_]['n']
                    val = dictionary[k_]['val']

                    # Construct k branch
                    if k not in data[i].keys():
                        data[i][k] = {}
                    # Construct l branch
                    if n not in data[i][k].keys():
                        data[i][k][n] = {}
                    # Construct l branch
                    if l not in data[i][k][n].keys():
                        data[i][k][n][l] = {}

                    if m in data[i][k][n][l].keys():
                        print('Index already defined: likely defined previously by father wavelet')

                    data[i][k][n][l][m] = val

        return data

    def wave_coefs(self, notflat: bool = False):
        # Rescale coefficient values
        ms = []
        for i in range(self.J + 1):
            if i == 0:
                ms.append(self.scaler[i] * self.grids[i])
            else:

                co = self.scaler[i] * self.grids[i]

                # Flatten / Dont flatten
                if notflat:
                    ms.append(co)
                else:
                    ms.append(co.flatten(1, 2))

        return ms

    def idwt_transform(self, idwt):
        coeffs = []
        for i in range(self.J + 1):
            coeffs.append(self.grids[i])

        yl = 0.
        yh = []

        for i in range(self.J + 1):
            if i == 0:
                yl = self.scaler[i] * coeffs[i]
            else:
                co = self.scaler[i] * coeffs[i]
                yh.append(co)

        fine = idwt((yl, yh))

        if self.what == 'spacetime':
            return fine + 1.
        return fine
    
    def yl_only(self):

        fine = self.scaler[0] * self.grids[0]
        if self.what == 'spacetime':
            return fine + 1.
        return fine


    def forward(self, pts, idwt):
        """Given a set of points sample the dwt transformed Kplanes and return features
        """
        plane = self.idwt_transform(idwt)
            
        signal = []
        
        if self.cachesig:
            signal.append(plane)
        
        # Sample features
        feature = (
            grid_sample_wrapper(plane, pts)
            .view(-1, plane.shape[1])
        )
        
        # visualize_grid_and_coords(plane, pts)


        self.signal = signal
        self.step += 1

        # Return multiscale features
        return feature


class WavePlaneField(nn.Module):
    def __init__(
            self,
            bounds,
            planeconfig,
            multires,
            use_rotation=False,
            is_opacity_grid=False,
    ):
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.is_opacity_grid = is_opacity_grid
        self.grid_config = [planeconfig]

        # 1. Init planes
        self.grids = nn.ModuleList()

        self.idwt = DWTInverse(wave='coif4', mode='periodization').cuda().float()

        self.cacheplanes = True
        self.is_waveplanes = True
        
        
        j, k = 0, 1
        for i in range(6):
            # We only need a grid for the space-time features
            if k == 3: 
                what = 'spacetime'
                res = [self.grid_config[0]['resolution'][j], self.grid_config[0]['resolution'][3]]
            else:
                what = 'space'
                res = [self.grid_config[0]['resolution'][j],
                    self.grid_config[0]['resolution'][k]]
            
            gridset = GridSet(
                what=what,
                resolution=res,
                J=self.grid_config[0]['wavelevel'],
                config={
                    'feature_size': self.grid_config[0]["output_coordinate_dim"],
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
                
        self.feat_dim = self.grid_config[0]["output_coordinate_dim"]

        init_plane = torch.empty([3, 3]).cuda()
        nn.init.uniform_(init_plane, a=-0.1, b=.1)

        if use_rotation:
            self.reorient_grid = nn.Parameter(init_plane, requires_grad=True)
        else:
            self.reorient_grid = None

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
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ], dtype=torch.float32)
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
            pts, self.grids, self.idwt, self.reorient_grid, True
        )


    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is not None:
            timestamps = (timestamps*2.)-1. # normalize timestamps between 0 and 1
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])

        return interpolate_features_MUL(
            pts, self.grids, self.idwt, self.reorient_grid, False
        )

    def get_opacity_vars(self, pts):
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, pts.shape[-1])
        return interpolate_features_MUL(
            pts, self.grids, self.idwt, self.reorient_grid, True
        )

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None, iterations: int = 0):

        return self.get_density(pts, timestamps)
