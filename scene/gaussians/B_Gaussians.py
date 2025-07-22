import torch
from scene.deformation import deform_network
from gaussian_renderer import render_motion_point_mask
from scene.gaussians.base import GaussianBase

class BackgroundGaussians(GaussianBase):
    
    def __init__(self, sh_degree, args, name='background'):
        
        deformation = deform_network(args, name=name)
        super().__init__(deformation, sh_degree, args, name=name)
    