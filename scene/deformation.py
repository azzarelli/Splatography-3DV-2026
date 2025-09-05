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
        self.canon_enc = nn.Sequential(nn.Linear(self.grid.feat_dim,net_size))
        self.background_canon_enc = nn.Sequential(nn.Linear(self.background_grid.feat_dim,net_size))
        self.background_deform_enc = nn.Sequential(nn.Linear(self.background_grid.feat_dim,net_size))
        
        self.background_pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))

        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.rot_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 4))
        
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 16*3))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 1))
    
        # Shared color, rotation and opacity canonical feature decoder
        self.shs_canon = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 16*3))
        self.rot_canon = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 4))
        self.opacity_canon = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))
        

    def forward_canon(self, pts, mask,return_type):
        if return_type == 'foreground':
            pts = pts[mask]
            feature = self.canon_enc(self.grid.space_only(pts))
            
            rotations = self.rot_canon(feature)
            opacity = self.opacity_canon(feature)
            shs = self.shs_canon(feature)
            
            return pts, rotations, opacity, shs
        
        elif return_type == 'background':
            pts = pts[~mask]
            feature = self.background_canon_enc(self.background_grid.space_only(pts))
            
            rotations = self.rot_canon(feature)
            opacity = self.opacity_canon(feature)
            shs = self.shs_canon(feature)
            
            return pts, rotations, opacity, shs
        
        elif return_type == 'full':
            pts_fg = pts[mask]
            feature_fg = self.canon_enc(self.grid.space_only(pts_fg))
            pts_bg = pts[~mask]
            feature_bg = self.background_canon_enc(self.background_grid.space_only(pts_bg))
            
            batch_features = torch.cat([feature_fg, feature_bg], dim=0)
            
            rotations = self.rot_canon(batch_features)
            opacity = self.opacity_canon(batch_features)
            shs = self.shs_canon(batch_features)
            
            return torch.cat([pts_fg, pts_bg], dim=0), rotations, opacity, shs
            
    def forward_fine(self, pts, time, mask, return_type):
        time = (time*2.)-1. # go from 0 to 1 to -1 to +1 for grid interp
        time = torch.zeros_like(pts[:, 0].unsqueeze(-1), dtype=torch.float, device=pts.device)
        if return_type == 'foreground':
            pts = pts[mask]
            time = time[mask]
            canon_feature = self.grid.space_only(pts)
            
            deformation_feature = self.deform_enc(self.grid(pts, time) * canon_feature)
            canon_feature = self.canon_enc(canon_feature)
            
            pts += self.pos_deform(deformation_feature)
            rotations = self.rot_canon(canon_feature) + self.rot_deform(deformation_feature)
            opacity = self.opacity_canon(canon_feature) + self.opacity_deform(deformation_feature)
            shs = self.shs_canon(canon_feature) + self.shs_deform(deformation_feature)
            
            return pts, rotations, opacity, shs
        
        elif return_type == 'background':
            pts = pts[~mask]
            time = time[~mask]

            canon_feature = self.background_grid.space_only(pts)
            
            deformation_feature = self.background_deform_enc(self.background_grid(pts, time) * canon_feature)
            canon_feature = self.background_canon_enc(canon_feature)

            
            pts += self.pos_deform(deformation_feature)
            rotations = self.rot_canon(canon_feature) + self.rot_deform(deformation_feature)
            opacity = self.opacity_canon(canon_feature) + self.opacity_deform(deformation_feature)
            shs = self.shs_canon(canon_feature) + self.shs_deform(deformation_feature)
            
            return pts, rotations, opacity, shs
        
        elif return_type == 'full':
            pts_fg = pts[mask]
            canon_feature_fg = self.grid.space_only(pts_fg)
            deformation_feature_fg = self.deform_enc(self.grid(pts_fg, time[mask]) * canon_feature_fg)
            canon_feature_fg = self.canon_enc(canon_feature_fg)
            
            fg_count = canon_feature_fg.shape[0]
            
            pts_bg = pts[~mask]
            canon_feature_bg = self.background_grid.space_only(pts_bg)
            deformation_feature_bg = self.background_deform_enc(self.background_grid(pts_bg, time[~mask]) * canon_feature_bg)
            canon_feature_bg = self.background_canon_enc(canon_feature_bg)
            
            batch_canon = torch.cat([canon_feature_fg, canon_feature_bg], dim=0)
            
            pts = torch.cat([pts_fg, pts_bg], dim=0)
            rotations = self.rot_canon(batch_canon)
            opacity = self.opacity_canon(batch_canon)
            shs = self.shs_canon(batch_canon)
            
            pts[:fg_count, ...] += self.pos_deform(deformation_feature_fg)
            pts[fg_count:, ...] += self.background_pos_deform(deformation_feature_bg)
            
            rotations[:fg_count, ...] += self.rot_deform(deformation_feature_fg)
            opacity[:fg_count, ...] += self.opacity_deform(deformation_feature_fg)
            shs[:fg_count, ...] += self.shs_deform(deformation_feature_fg)
            
            return pts, rotations, opacity, shs
            

    def forward(self,
            xyz,
            time,
            target_mask,
            stage,
            return_type
        ):
        if stage == 'coarse':
            pts, rotations, opacity, shs = self.forward_canon(xyz, target_mask, return_type)
        elif stage == 'fine':
            pts, rotations, opacity, shs = self.forward_fine(xyz, time, target_mask, return_type)
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

    def forward(self, point=None,time=None,target_mask=None, stage=None, return_type='full'):

        return  self.deformation_net(
            point,
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
            