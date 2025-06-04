import torch
import torch.nn as nn
import torch.nn.functional as F

from submodules.pytorch_wavelets_.dwt.transform2d import DWTForward,DWTInverse


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, sample_method:str='bilinear', align_corners: bool = True) -> torch.Tensor:
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
        mode=sample_method, padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def grid_sample_wrapper_temporal(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    grid_sampler = F.grid_sample

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]

    # Grid is range -1 to 1 and is dependant on the resolution
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=False,
        mode='nearest', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp



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
            sample_method:str = 'bilinear'
    ):
        super().__init__()

        self.what = what
        self.config = config
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
        self.current_J = J
        self.sample_method = sample_method

        # Initialise a signal to DWT into our initial Wave coefficients
        dwt = DWTForward(J=J, wave=config['wave'], mode=config['wave_mode']).cuda()
        
        self.idwt = DWTInverse(wave='coif4', mode='periodization').cuda().float()
        
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

    def update_J(self):
        pass
        # if self.current_J <= self.J:
        #     self.current_J += 1

    def idwt_transform(self):
        yl = 0.
        yh = []
        for i in range(self.J + 1):
            if i == 0:
                yl = self.scaler[i] * self.grids[i]
            else:
                co = self.scaler[i] * self.grids[i]
                yh.append(co)

        # yh_rev = yh[::-1]
        # yh_rev = yh_rev[:self.current_J]
        # yh = yh_rev[::-1]
        fine = self.idwt((yl, yh))

        if self.what == 'spacetime':
            return fine + 1.
        return fine 

    def forward(self, pts):
        """Given a set of points sample the dwt transformed Kplanes and return features
        """
        plane = self.idwt_transform()
            
        signal = []
        
        if self.cachesig:
            signal.append(plane)
        
        # Sample features
        feature = (
            grid_sample_wrapper(plane, pts, sample_method=self.sample_method)
            .view(-1, plane.shape[1])
        )        

        self.signal = signal
        self.step += 1

        # Return multiscale features
        return feature