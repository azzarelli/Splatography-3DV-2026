from typing import Optional

from submodules.pytorch_wavelets_.dwt.transform2d import DWTInverse, DWTForward

from utils.kplane_utils import normalize_aabb, GridSet

import torch
import torch.nn as nn
import torch.nn.functional as F

def interpolate_features(pts: torch.Tensor, planes, idwt):
    """Generate features for each point
    """
    # time m feature
    interp_1 = 1.
    q,r = 0,1
    for i in range(3):
        
        feature = planes[i](pts[..., (q, r)], idwt)
        interp_1 = interp_1 * feature
        
        r +=1
        if r == 3:
            q = 1
            r = 2
    return interp_1


class TriPlaneField(nn.Module):
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
        for i in range(3):
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
            if k == 3:
                j = 1
                k = 2
                

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

    def get_grids(self, notflat: bool = False):
        """Return the grids as a list of parameters for regularisation
        """
        ms_planes = []
        for i in range(3):

            ms_planes.append(self.grids[i].signal[0])

        return ms_planes


    def forward(self,
                pts: torch.Tensor):
        
        pts = normalize_aabb(pts, self.aabb)

        return interpolate_features(
            pts, self.grids, self.idwt)
