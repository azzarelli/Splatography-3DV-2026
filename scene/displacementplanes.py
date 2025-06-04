from typing import Optional

from submodules.pytorch_wavelets_.dwt.transform2d import DWTInverse

from utils.kplane_utils import grid_sample_wrapper, normalize_aabb, GridSet

import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_features_MUL(data, kplanes):
    """Generate features for each point
    """
    # time m feature
    spacetime = 1.

    # q,r are the coordinate combinations needed to retrieve pts
    coords = [[3,0], [3,1], [3,2]]
    for i in range(len(coords)):
        q,r = coords[i]
        feature = kplanes[i](data[..., (q, r)])
        feature = feature.view(-1,feature.shape[-1])
        spacetime = spacetime * feature

    return spacetime
   

class DisplacementField(nn.Module):
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
        
        for i in range(3):
            what = 'spacetime-disp'
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
                cachesig=self.cacheplanes,
                sample_method='nearest'
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

    def waveplanes_list(self):
        planes = []
        for grid in self.grids:
            planes.append(grid.grids)
        return planes
    
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

    def forward(self, pts, time, coarse=False):
        """
            Notes:
                - to visualize samples and projection use display_projection(pts, cov6)
                    - you can modify the constants K_a and K_b to see that the samples get plotted closer to the edge
        """
        time = (time*2.)-1. # go from 0 to 1 to -1 to +1 for grid interp
                
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat([pts.view(-1, 3), time], dim=-1)
        
        feature_A = interpolate_features_MUL(pts, self.grids)
        
        if coarse == False:
            # shift one 
            time_step = 1./ (2.*self.grid_config['resolution'][1])
            if pts[-1, 0] + time_step > 1. :
                feature_B = feature_A
                pts[-1, :] = pts[-1, :] - time_step

                feature_A = interpolate_features_MUL(pts, self.grids)
            else:
                pts[-1, :] = pts[-1, :] - time_step
                feature_B = interpolate_features_MUL(pts, self.grids)
    

        else:
            feature_B = None
        return feature_A, feature_B
