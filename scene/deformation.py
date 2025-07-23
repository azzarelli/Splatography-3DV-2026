import torch
import torch.nn as nn
import torch.nn.init as init
from scene.waveplanes import WavePlaneField

from scene.triplanes import TriPlaneField

import time

from torch_cluster import knn_graph
from utils.general_utils import strip_symmetric, build_scaling_rotation

class Deformation(nn.Module):
    def __init__(self, W=256, args=None, name='foreground'):
        super(Deformation, self).__init__()
        self.W = W
        self.name=name
        print(name)
        if name == 'foreground':
            self.grid = WavePlaneField(args.bounds, args.target_config, name)
        elif name == 'background':
            self.grid = WavePlaneField(args.bounds, args.scene_config, name)
        else:
            raise 'Requires specified name for background/foreground'

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

        
    def set_aabb(self, xyz_max, xyz_min):
        self.grid.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        # Prep features for decoding
        net_size = self.W
        self.spacetime_enc = nn.Sequential(nn.Linear(self.grid.feat_dim,net_size))
        self.spacecol_enc = nn.Sequential(nn.Linear(self.grid.feat_dim,net_size))
        
        self.pos_coeffs = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        
        if self.name == 'foreground':
            self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 4))
            self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 16*3))
        
    def query_spacetime(self, pts, time, covariances, mask=None):
        
        if self.name == 'foreground':
            space, spacetime, coltime = self.grid(pts, time, covariances)
            st = self.spacetime_enc(space * spacetime) # TODO: Different encoders for color and space time? Its only one layer though
            ct = self.spacecol_enc(space * coltime)
        elif self.name == 'background':
            space_b, spacetime_b, _ = self.grid(pts, time, covariances)
            st = self.spacetime_enc(space_b * spacetime_b)
            ct = None
        return st, ct

    def forward(self, rays_pts_emb, rotations_emb, scale_emb, shs_emb, h_emb, time_emb):
        time_emb = torch.full_like(scale_emb[:,0].unsqueeze(-1), time_emb).to(rays_pts_emb.device)
        covariances = self.covariance_activation(scale_emb, 1., rotations_emb)
        dyn_feature, color_feature = self.query_spacetime(rays_pts_emb, time_emb, covariances)
        
        pts = rays_pts_emb + self.pos_coeffs(dyn_feature)

        if self.name == 'background':
            return pts, rotations_emb, None, shs_emb 

        rotations = None
        shs = None
        opacity = None
        if shs_emb is not None:
            rotations = rotations_emb + self.rotations_deform(dyn_feature)
            shs = shs_emb + self.shs_deform(color_feature).view(-1, 16, 3)
            
            if self.name == 'foreground' and h_emb is not None:
                t = time_emb[0:1].squeeze(0)
                opacityh, opacityw, opacitymu = h_emb
                w = (opacityw**2)
                mu = torch.sigmoid(opacitymu)
                opacity = torch.sigmoid(opacityh) * torch.exp(-w * (t-mu)**2)
            elif h_emb is not None:
                opacity = h_emb
            

        return pts, rotations, opacity, shs
    
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
    def __init__(self, args, name='foreground') :
        super(deform_network, self).__init__()
        net_width = args.net_width

        self.deformation_net = Deformation(W=net_width,  args=args, name=name)

        self.apply(initialize_weights)

    def forward(self, point, rotations=None, scales=None, shs=None, time=None, h_emb=None):

        return  self.deformation_net(
            point,
            rotations,
            scales,
            shs,
            h_emb,
            time, 
        )

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
            