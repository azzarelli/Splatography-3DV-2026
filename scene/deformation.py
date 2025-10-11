import torch
import torch.nn as nn
import torch.nn.init as init
from scene.waveplanes import WavePlaneField

from scene.triplanes import TriPlaneField

import time

from torch_cluster import knn_graph
from utils.general_utils import strip_symmetric, build_scaling_rotation

# Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
RGB2XYZ = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=torch.float).cuda()  # shape (3, 3)

XYZ2RGB = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=torch.float).cuda()  # shape (3, 3)


def rgb_to_xyz(rgb):
    threshold = 0.04045
    rgb_linear = torch.where(
        rgb <= threshold,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    
    xyz = torch.matmul(rgb_linear, RGB2XYZ.T)

    return xyz

def xyz_to_rgb(xyz):
    rgb_linear = torch.matmul(xyz, XYZ2RGB.T)

    threshold = 0.0031308
    rgb =  torch.where(
        rgb_linear <= threshold,
        12.92 * rgb_linear,
        1.055 * (rgb_linear.clamp(min=1e-8) ** (1/2.4)) - 0.055
    )

    return rgb.clamp(0.0, 1.0) 

class Deformation(nn.Module):
    def __init__(self, W=256, args=None):
        super(Deformation, self).__init__()
        self.W = W
        self.grid = WavePlaneField(args.bounds, args.target_config)
        # self.color_grid = WavePlaneField(args.bounds, args.scene_config)
        self.background_grid = WavePlaneField(args.bounds, args.scene_config)
        # self.fine_grid = WavePlaneField(args.bounds, args.scene_config, rotate=True)

        # self.target_grid.aabb = None

        self.args = args

        self.ratio=0
        self.create_net()
        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        # inputs scaling, scalingmod=1.0, rotation
        self.covariance_activation = build_covariance_from_scaling_rotation

        
    def set_aabb(self, xyz_max, xyz_min, grid_type='target'):
        if grid_type=='target':
            self.grid.set_aabb(xyz_max, xyz_min)
        elif grid_type=='background':
            self.background_grid.set_aabb(xyz_max, xyz_min)
    
    
    def create_net(self):
        # Prep features for decoding
        net_size = self.W
        
        self.deform_enc = nn.Sequential(nn.Linear(self.grid.feat_dim,net_size))
        self.background_deform_enc = nn.Sequential(nn.Linear(self.background_grid.feat_dim,net_size))
        
        self.background_pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.rot_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 4))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 16*3))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 1))
    

    def forward_canon(self, pts, rotations, colors, opacity, mask, return_type):
        pts = pts + 0.
        rotations = rotations + 0.
        opacity = opacity + 0.
        colors = colors + 0.
        if return_type == 'foreground':
            pts = pts[mask]
            
            rotations = rotations[mask] 
            opacity = opacity[mask]
            shs = colors[mask]
            
        elif return_type == 'background':
            pts = pts[~mask]
            
            rotations = rotations[~mask] 
            opacity = opacity[~mask]
            shs = colors[~mask]
            
        elif return_type == 'full':
            shs=colors

        opacity = torch.sigmoid(opacity[:,0]).unsqueeze(-1)            
        
        return pts, rotations, opacity, shs
            
    def forward_fine(self, pts, rotations, colors, opacity, time, mask, return_type):
        t = (time*2.)-1. # go from 0 to 1 to -1 to +1 for grid interp
        time = torch.zeros_like(pts[:, 0].unsqueeze(-1), dtype=torch.float, device=pts.device)
        time += t
        pts = pts + 0.
        rotations = rotations + 0.
        opacity = opacity + 0.
        colors = colors + 0.
        
        if return_type == 'foreground':
            pts = pts[mask]
            time = time[mask]
            canon_feature = self.grid.space_only(pts)
            dynamic_feature = self.grid(pts, time)
            deformation_feature = self.deform_enc(dynamic_feature* canon_feature)

            pts += self.pos_deform(deformation_feature)
            rotations = rotations[mask] + self.rot_deform(deformation_feature)
            opacity = opacity[mask, 0].unsqueeze(-1) + self.opacity_deform(deformation_feature)
            shs = colors[mask] + self.shs_deform(deformation_feature).view(-1, 16, 3)
            
        
        elif return_type == 'background':
            pts = pts[~mask]
            time = time[~mask]

            canon_feature = self.background_grid.space_only(pts)
            dynamic_feature = self.background_grid(pts, time)
            deformation_feature = self.background_deform_enc(dynamic_feature * canon_feature)
            
            pts += self.background_pos_deform(deformation_feature)
            rotations = rotations[~mask]
            opacity = opacity[~mask,0].unsqueeze(-1)
            shs = colors[~mask]            
        
        elif return_type == 'full':
            shs = colors
            opacity = opacity[:,0].unsqueeze(-1)

            pts_fg = pts[mask]
            deformation_feature_fg = self.deform_enc(self.grid(pts_fg, time[mask]) * self.grid.space_only(pts_fg))
       
            pts_bg = pts[~mask]
            deformation_feature_bg = self.background_deform_enc(self.background_grid(pts_bg, time[~mask]) * self.background_grid.space_only(pts_bg))

            # Positional deformations
            pts[mask] += self.pos_deform(deformation_feature_fg)
            pts[~mask] += self.background_pos_deform(deformation_feature_bg)
            
            # Other Fg deformations
            rotations[mask] += self.rot_deform(deformation_feature_fg)
            opacity[mask] += self.opacity_deform(deformation_feature_fg)
            shs[mask] += self.shs_deform(deformation_feature_fg).view(-1, 16, 3)
        
        opacity = torch.sigmoid(opacity)
        return pts, rotations, opacity, shs
    
       

    def forward(self,
            xyz,
            rotations,colors,opacity,
            time,
            target_mask,
            stage,
            return_type
        ):
        if stage == 'coarse':
            pts, rotations, opacity, shs = self.forward_canon(xyz,rotations,colors,opacity, target_mask, return_type)
        elif stage == 'fine':
            # LERP time between keyframes
            kf = [0.0, 0.1806020066889632, 0.36789297658862874, 0.8896321070234113, 1.0]
            t_hex = [ 0., 0.25, 0.5, 0.75]
            t_interval = 0.25
            # Equivalent to 
            #      0    0.25,                 0.5,              0.75,                 1
            for idx in range(len(kf)-1):
                if time > kf[idx] and time < kf[idx+1]:
                    time = (time - kf[idx])/(kf[idx+1]-kf[idx])*t_interval + t_hex[idx]
                    
            pts, rotations, opacity, shs = self.forward_fine(xyz,rotations,colors,opacity, time, target_mask, return_type)
        else:
            raise f"No implement stage {stage} in deformation forward function"
        
        return pts, rotations, opacity, shs
        # if target_mask is None: # Sample features at the 
        #     shs = shs_emb # + self.shs_deform(color_feature).view(-1, 16, 3)
            
        #     pts = rays_pts_emb # + self.pos_coeffs(dyn_feature)
        #     rotations = rotations_emb # + self.rotations_deform(dyn_feature)
            
        #     opacity = torch.sigmoid(h_emb[:,0]).unsqueeze(-1)
        #     w = (h_emb[:,1]**2).unsqueeze(-1)
        #     mu = torch.sigmoid(h_emb[:,2]).unsqueeze(-1)
        #     t = time_emb[0:1].squeeze(0)
        #     feat_exp = torch.exp(-w * (t-mu)**2)
        #     opacity = feat_exp # h_emb[target_mask] * feat_exp
        #     return pts, rotations, opacity, shs, None

        # covariances = self.covariance_activation(scale_emb, 1., rotations_emb)
        # dyn_feature, color_feature, background_feature = self.query_spacetime(rays_pts_emb,time_emb, covariances, target_mask)
        
        # # Rotation
        # rotations = rotations_emb + 0.
        # rotations[target_mask] += self.rotations_deform(dyn_feature)
        
        # shs = shs_emb + 0.
        # shs[target_mask] += self.shs_deform(dyn_feature).view(-1, 16, 3)
         
        # # Position
        # pts = rays_pts_emb + 0. #.clone()        
        # pts[target_mask] += self.pos_coeffs(dyn_feature)
        # pts[~target_mask] += self.background_pos_coeffs(background_feature)
        
        # # Opacity
        # opacity = torch.sigmoid(h_emb[:,0]).unsqueeze(-1)
        # w = (h_emb[target_mask,1]**2).unsqueeze(-1)
        # mu = torch.sigmoid(h_emb[target_mask,2]).unsqueeze(-1)
        
        # t = time_emb[0:1].squeeze(0)
        # feat_exp = torch.exp(-w * (t-mu)**2)
        # opacity = opacity.clone()
        # opacity[target_mask] = feat_exp # h_emb[target_mask] * feat_exp
 
        # return pts, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name and 'background' not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_background_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name and 'background' in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name and 'background' not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_background_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name and 'background' in name:
                parameter_list.append(param)
        return parameter_list
    

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width

        self.deformation_net = Deformation(W=net_width,  args=args)

        self.apply(initialize_weights)

    def forward(self, point=None, rotations=None, colors=None, opacity=None,time=None,target_mask=None, stage=None, return_type='full'):

        return  self.deformation_net(
            point,
            rotations,colors,opacity,
            time,
            target_mask,
            stage,
            return_type
        )

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 
    
    def get_background_mlp_parameters(self):
        return self.deformation_net.get_background_mlp_parameters() 
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    
    def get_background_grid_parameters(self):
        return self.deformation_net.get_background_grid_parameters()
    
    def get_dyn_coefs(self, xyz, scale):
        return self.deformation_net.get_dx_coeffs(xyz, scale)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
            