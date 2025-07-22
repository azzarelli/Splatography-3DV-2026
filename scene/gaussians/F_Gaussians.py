import torch
from scene.deformation import deform_network
from gaussian_renderer import render_motion_point_mask
from scene.gaussians.base import GaussianBase

class ForegroundGaussians(GaussianBase):
    
    def __init__(self, sh_degree, args, name='foreground'):
        
        deformation = deform_network(args, name=name)
        super().__init__(deformation, sh_degree, args, name=name)
    
    def get_position_changes(self, time):
        scales = self.get_scaling_with_3D_filter
        
        means3D, _, _, _  = self._deformation(
            point=self._xyz, 
            rotations=self._rotation,
            scales=scales, 
            shs=None, 
            h_emb=None,
            time=time
        )
        
        return means3D
    
    def dupelicate(self):
        new_xyz = self._xyz+ torch.rand_like(self._xyz)*0.005
        new_features_dc = self._features_dc
        new_features_rest = self._features_rest
        new_scaling = self._scaling
        new_rotation = self._rotation
        new_opacitiesh = self._opacityh
        new_opacitiesw = self._opacityw
        new_opacitiesmu = self._opacitymu
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacitiesh, new_opacitiesw,
                                   new_opacitiesmu, new_scaling, new_rotation)

    @torch.no_grad()
    def dynamic_dupelication(self):
        """Duplicate points with highly dynamic motion - maybe the top 10% of points with the largest motions?
        """
        
        selected_pts_mask = render_motion_point_mask(self)
        
        new_xyz = self._xyz[selected_pts_mask] + torch.rand_like(self._xyz[selected_pts_mask])*0.005
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_opacitiesh = self._opacityh[selected_pts_mask]
        new_opacitiesw = self._opacityw[selected_pts_mask]
        new_opacitiesmu = self._opacitymu[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacitiesh,new_opacitiesw,new_opacitiesmu, new_scaling, new_rotation)
          