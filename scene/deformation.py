import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField

from scene.waveplanes import HexPlaneField as WavePlaneField

from scene.grid import DenseGrid

import time
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.no_grid = args.no_grid

        self.opacity_grid = False
        if args.use_waveplanes:
            print("[Using WavePlanes]")
            self.grid = WavePlaneField(args.bounds, args.kplanes_config, args.multires, use_rotation=args.plane_rotation_correction)
            # self.opacity_grid = WavePlaneField(args.bounds, args.kplanes_config['opacity_config'], args.multires, use_rotation=args.plane_rotation_correction, is_opacity_grid=True)
        else:
            self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    def create_net(self):
        # Prep features for decoding
        net_size = self.W
        insize = self.grid.feat_dim 
        self.feature_out = nn.Sequential(nn.Linear(insize,net_size))

        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 4))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 16*3))

        # Same for opacity grid
        self.opacity_featu_out = nn.Sequential(nn.Linear(insize, net_size))
      
        # self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_h = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_w = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_mu = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))


    def query_time(self, rays_pts_emb, time_emb, iterations):
        
        st, sp = self.grid(rays_pts_emb[:,:3], time_emb[:,:1], iterations)
        # Querying the feature grids
        st_features = self.feature_out(
            st
        )
        
        sp_features = self.opacity_featu_out(
            sp
        )
        
        return st_features, sp_features
    
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None,shs_emb=None, time_feature=None, time_emb=None, iterations=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:

            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, shs_emb, time_feature, time_emb, iterations)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    
    def foward_opac(self, xyz):
        feature =  self.opacity_featu_out(
            self.grid.get_opacity_vars(xyz)
        )
        return self.opacity_w(feature), torch.sigmoid(self.opacity_h(feature)), torch.sigmoid(self.opacity_mu(feature))
        
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, shs_emb, time_feature, time_emb, iteration):
        
        hidden, hidden_opac = self.query_time(rays_pts_emb, time_emb, iteration)

        # Change in position
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:

            dx_start = self.pos_deform(hidden)
            pts = rays_pts_emb[:,:3] + dx_start

        # Change in scale
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds

        # Change in rotation
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr
        
        # Change in opacity
        if self.args.no_do :
            w = None
        else:

            # Lets start with the simple part - 
            # Lets decode the main feature into width and height features, seperately
            w = self.opacity_w(hidden_opac)
            h = torch.sigmoid(self.opacity_h(hidden_opac)) #(torch.cos(self.opacity_h(hidden_opac))+1.)/2. # between 0 and 1
            # mu = (torch.cos(self.opacity_mu(hidden_opacity))+1.)/2. # between 0 and 1
            mu = torch.sigmoid(self.opacity_mu(hidden_opac)) #opacity_emb #self.opacity_mu(hidden_opac) #(torch.cos(self.opacity_mu(hidden_opac))+1.)/2.# between 0 and 1 (torch.cos(opacity_emb)+1.)/2. # 

            # Now for the temporal opacity function
            #  y = h exp(-(w^2)|x-u|^2)
            opacity = h * torch.exp(-(w**2)*((time_emb- mu)**2))

        
        # Change in color        
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb + dshs

        if w is not None:
            return pts, scales, rotations, opacity, shs

        else:
            return pts, scales, rotations, opacity, shs
    
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
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, shs=None, times_sel=None, iterations=None):
        return self.forward_dynamic(point, scales, rotations, shs, times_sel, iterations)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None,shs=None, times_sel=None, iterations=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net(point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                shs,
                                                None,
                                                times_sel, iterations)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

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