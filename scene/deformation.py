import torch
import torch.nn as nn
import torch.nn.init as init
from scene.waveplanes import WavePlaneField

from scene.triplanes import TriPlaneField
from scene.displacementplanes import DisplacementField
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
        self.grid = WavePlaneField(args.bounds, args.scene_config)
        self.displacement_grid = DisplacementField(args.bounds, args.scene_config)
        self.background_grid = WavePlaneField(args.bounds, args.target_config)
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

        
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min, grid_type='target'):
        if grid_type=='target':
            self.grid.set_aabb(xyz_max, xyz_min)
            self.displacement_grid.set_aabb(xyz_max, xyz_min)
        elif grid_type=='background':
            self.background_grid.set_aabb(xyz_max, xyz_min)
    
    
    def create_net(self):
        # Prep features for decoding
        net_size = self.W
        insize = self.grid.feat_dim 
        
        self.spacetime_enc = nn.Sequential(nn.Linear(insize,net_size))
        self.background_spacetime_enc = nn.Sequential(nn.Linear(insize,net_size))
        self.color_enc = nn.Sequential(nn.Linear(insize,net_size))
        self.coefenc = nn.Sequential(nn.Linear(insize,net_size))

        self.background_pos_coeffs = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.pos_coeffs = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 4))
        self.rgb_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.rgb_deform2 = nn.Sequential(nn.ReLU(),nn.Linear(4, net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.scale_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 3))

        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 16*3))
    
    
        # intial rgb, temporal rgb, rotation
        self.rgb_decoder = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 1))

    def update_wavelevel(self):
        self.grid.update_J()

    def query_spacetime(self, rays_pts_emb, time, covariances, mask=None):
        
        if mask is not None:
            space, spacetime = self.grid(rays_pts_emb[mask,:3], time[mask,:], covariances[mask])
            std_A, std_B = self.displacement_grid(rays_pts_emb[mask,:3], time[mask,:])
            std_B = self.coefenc(space*std_B)

            space_b, spacetime_b = self.background_grid(rays_pts_emb[~mask,:3], time[~mask,:], covariances[~mask])
            st_b = self.background_spacetime_enc(space_b * spacetime_b)

        else:
            space, spacetime = self.grid(rays_pts_emb[:,:3], time, covariances)
            std_A, std_B = self.displacement_grid(rays_pts_emb[:,:3], time, coarse=True)
            st_b = None

        st = self.spacetime_enc(space * spacetime)
        std_A = self.coefenc(space*std_A)
        return st, std_A, std_B, st_b # * spacetime # *  sp_fine_features# or maybe multiply and use scale to modulate the sp_fine e.g. low scale high influence

    def query_spacetheta(self, rays_pts_emb, angle, input_feature, mask=None):
        
        if mask is not None:
            angle_feature = self.grid.theta(rays_pts_emb[mask,:3], angle)
        else:
            angle_feature = self.grid.theta(rays_pts_emb[:,:3], angle)

        sttheta = self.color_enc(input_feature*angle_feature)
        return sttheta
    

    def forward(self,rays_pts_emb, rotations_emb, scale_emb, shs_emb, view_dir, time_emb, h_emb, target_mask):
        # Features
        covariances = self.covariance_activation(scale_emb, 1., rotations_emb)
        dyn_feature, disp_feature_A, disp_feature_B, background_feature = self.query_spacetime(rays_pts_emb,time_emb, covariances, target_mask)
        
        if target_mask is None: # Sample features at the 
            shs = shs_emb + 0.
            shs[:, 1:] = shs[:, 1:] + self.shs_deform(dyn_feature).view(-1, 15, 3)
            # pts = rays_pts_emb + 0.
            pts = rays_pts_emb + self.pos_coeffs(disp_feature_A)            
            # pts = rays_pts_emb + self.pos_coeffs(dyn_feature)
            rotations = rotations_emb + self.rotations_deform(dyn_feature)
            
            opacity = torch.sigmoid(h_emb[:,0]).unsqueeze(-1)
            w = (h_emb[:,1]**2).unsqueeze(-1)
            mu = torch.sigmoid(h_emb[:,2]).unsqueeze(-1)
            t = time_emb[0:1].squeeze(0)
            feat_exp = torch.exp(-w * (t-mu)**2)
            opacity = feat_exp # h_emb[target_mask] * feat_exp
            return pts, rotations, opacity, shs, None

        # Rotation
        rotations = rotations_emb + 0.
        rotations[target_mask] += self.rotations_deform(dyn_feature)
        
        # Color
        # Get norm from rotation and scales
        # norms = rotated_softmin_axis_direction(rotations[target_mask], scale_emb[target_mask])
        # norms = norms / norms.norm()
        # cos_theta = torch.clamp(torch.matmul(norms, view_dir.cuda()), -1.0, 1.0).unsqueeze(-1)
        # col_feature = self.query_spacetheta(rays_pts_emb, cos_theta, spacetime_feature, target_mask)
    
        # xyz = rgb_to_xyz(shs_emb)
        # xyz[target_mask, 1] += self.rgb_decoder(dyn_feature).squeeze(-1) 
        # shs = xyz_to_rgb(xyz)
        shs = shs_emb + 0.
        # shs[target_mask] += self.rgb_deform2(torch.cat([self.rgb_deform(dyn_feature), cos_theta], dim=-1)) #self.rgb_decoder(dyn_feature).squeeze(-1) 
        shs[target_mask, 1:] = shs_emb[target_mask, 1:] + self.shs_deform(dyn_feature).view(-1, 15, 3)
        
        # Opacity
        opacity = torch.sigmoid(h_emb[:,0]).unsqueeze(-1)
        w = (h_emb[target_mask,1]**2).unsqueeze(-1)
        mu = torch.sigmoid(h_emb[target_mask,2]).unsqueeze(-1)
        
        t = time_emb[0:1].squeeze(0)
        feat_exp = torch.exp(-w * (t-mu)**2)
        opacity = opacity.clone()
        opacity[target_mask] = feat_exp # h_emb[target_mask] * feat_exp
        
        # Position
        # We want to interpollate the two positions A and B from our feature space
        A = rays_pts_emb[target_mask] + self.pos_coeffs(disp_feature_A)
        B = rays_pts_emb[target_mask] + self.pos_coeffs(disp_feature_B)
        res = (2.* self.displacement_grid.grid_config['resolution'][1])
        interval = 1. /res
        t_index = torch.floor(t*res).float()
        t_value = t_index*interval
        t_delta = t-t_value
        pts = rays_pts_emb + 0.
        pts[target_mask] = (1.-t_delta)*A + (t_delta)*B

        # Non-linear and uninterpretable
        pts[~target_mask] += self.background_pos_coeffs(background_feature)
        
        
        # scales = scale_emb + 0.
        # scales[target_mask] += self.scale_deform(dyn_feature)
        if False:
            w_np = w.cpu().numpy().flatten()
            h_np = h.cpu().numpy().flatten()
            mu_np = mu.cpu().numpy().flatten()
            opacity_np = opacity.cpu().numpy().flatten()

            # pass
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid of subplots

            # Plot each histogram in its own axis
            axes[0, 0].hist(h_np, bins=30, color='blue', edgecolor='black')
            axes[0, 0].set_title('Histogram of h')

            axes[0, 1].hist(w_np, bins=100, color='green', edgecolor='black')
            axes[0, 1].set_title('Histogram of w')

            axes[1, 0].hist(mu_np, bins=30, color='red', edgecolor='black')
            axes[1, 0].set_title('Histogram of mu')

            axes[1, 1].hist(opacity_np, bins=30, color='purple', edgecolor='black')
            axes[1, 1].set_title('Histogram of opacity')

            # Adjust layout
            plt.tight_layout()
            plt.show()
            exit()

        return pts, rotations, opacity, shs, None
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list

SQRT_PI = torch.sqrt(torch.tensor(torch.pi))
def gaussian_integral(w):
    """Returns high weight (0 to 1) for solid materials
    
        Notes on optimization:
            The initial integral for a gaussian from 0 to 1 is (SQRT_PI / (2 * w)) * (erf_term_1 - erf_term_2), 
            with the center at 0. Instead, to be more precise/sensitive and faster we evaluate the integral between
            -1 and 1 with the gaussian centered at mu=0, as SQRT_PI*erf1/w. This reduces the complexity of
            the integral
    """
    SQRT_PI = torch.sqrt(torch.tensor(torch.pi, dtype=w.dtype, device=w.device))
    EPS = 1e-8  # for numerical stability
    return (SQRT_PI / (w + EPS)) * torch.erf(w)

import torch.nn.functional as F
def quaternion_rotate(q, v):
    q_vec = q[:, :3]
    q_w = q[:, 3].unsqueeze(1)
    t = 2 * torch.cross(q_vec, v, dim=1)
    return v + q_w * t + torch.cross(q_vec, t, dim=1)

def rotated_softmin_axis_direction(r, s, temperature=10.0):
    # s: (N, 3), we want the direction of the smallest abs scale
    abs_s = torch.abs(s)

    # Step 1: Compute softmin weights (lower abs(s) => higher weight)
    weights = F.softmax(-abs_s * temperature, dim=1)  # (N, 3)

    # Step 2: Basis axes: x, y, z
    basis = torch.eye(3, device=s.device).unsqueeze(0)  # (1, 3, 3)

    # Step 3: Weighted sum of basis vectors
    soft_axis = torch.bmm(weights.unsqueeze(1), basis.repeat(s.size(0), 1, 1)).squeeze(1)  # (N, 3)

    # Step 4: Rotate the direction
    rotated = quaternion_rotate(r, soft_axis)  # (N, 3)

    return rotated        

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width

        self.deformation_net = Deformation(W=net_width,  args=args)

        self.apply(initialize_weights)

    def forward(self, point, rotations=None, scales=None, shs=None,view_dir=None, times_sel=None, h_emb=None, iteration=None, target_mask=None):

        return  self.deformation_net(
            point,
            rotations,
            scales,
            shs,
            view_dir,
            times_sel, 
            h_emb=h_emb, 
            target_mask=target_mask
        )
        
    
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb

    def update_wavelevel(self):
        self.deformation_net.update_wavelevel()

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    
    def get_dyn_coefs(self, xyz, scale):
        return self.deformation_net.get_dx_coeffs(xyz, scale)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
            
def poc_fre(input_data,poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb