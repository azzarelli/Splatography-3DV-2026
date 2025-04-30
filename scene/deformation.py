import torch
import torch.nn as nn
import torch.nn.init as init
from scene.waveplanes import WavePlaneField

from scene.triplanes import TriPlaneField

import time

from torch_cluster import knn_graph


class Deformation(nn.Module):
    def __init__(self, W=256, args=None):
        super(Deformation, self).__init__()
        self.W = W
        self.grid = WavePlaneField(args.bounds, args.scene_config)

        # self.target_grid.aabb = None

        self.args = args

        self.ratio=0
        self.create_net()
        
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        self.grid.set_aabb(xyz_max, xyz_min)

    
    def create_net(self):
        # Prep features for decoding
        net_size = self.W
        insize = self.grid.feat_dim 
        
        self.space_enc = nn.Sequential(nn.Linear(insize,net_size))
        self.spacetime_enc = nn.Sequential(nn.Linear(insize,net_size))

        self.pos_coeffs = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 9))
        
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 8))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 3))
        self.shs_base = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 16*3))
      
        # self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_h = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_w = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_mu = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))


    def query_time(self, rays_pts_emb,scale_emb, time_emb, iterations, mask=None):
        # Given point scales we should model the surrounding features
        # we need to project the point and it's scale onto the xy, xz, yz planes - hence we need to generate points based off the aabb
        # scale_emb
        
        # Sample target points
        sp = self.grid(rays_pts_emb[mask,:3], time_emb[mask,:1], scale_emb[mask, :3])
        sp_features = self.space_enc(sp)
        # st_features = self.spacetime_enc(st)
        return sp_features

    def get_dx_coeffs(self, xyz):
        return self.pos_coeffs(self.space_enc(self.grid.get_opacity_vars(xyz[:,:3])))

    def forward(self,rays_pts_emb, rotations_emb, scale_emb, shs_emb, time_emb, iteration, h_emb, target_mask):

        sp_features = self.query_time(rays_pts_emb,scale_emb, time_emb, iteration, target_mask)
        
        pts = rays_pts_emb + 0. #.clone()
        
        # 3rd order Bezier deformation for target points
        dx_coeffs = self.pos_coeffs(sp_features).view(-1, 3, 3) # N, 3, 3 
        t = time_emb[0:1].squeeze(0)
        mint = 1. - t
        pts[target_mask] += 3. * (mint**2) * t*dx_coeffs[:,0] + 3. * mint * (t**2) * dx_coeffs[:, 1] + (t**3)*dx_coeffs[:, 2]

        if h_emb is not None:
            opacity = torch.sigmoid(h_emb[:,0]).unsqueeze(-1)
            w = (h_emb[target_mask,1]**2).unsqueeze(-1)
            mu = torch.sigmoid(h_emb[target_mask,2]).unsqueeze(-1)
            
            feat_exp =  torch.exp(-w * (t-mu)**2)
            opacity = opacity.clone()
            opacity[target_mask] = feat_exp # h_emb[target_mask] * feat_exp
        else:
            opacity = None
            
        # Change in rotation
        rotations = rotations_emb + 0.
        rotation_coeffs = self.rotations_deform(sp_features).view(-1, 2, 4)

        rotations[target_mask] += 2. * mint*t*rotation_coeffs[:,0] + (t**2)*rotation_coeffs[:, 1]
                
        if shs_emb is not None:
            shs = shs_emb + 0.
            base = self.shs_base(sp_features)
            shs_coeffs = self.shs_deform(sp_features).view(-1, 3, 1)
            dshs = base*feat_exp* (3. * (mint**2) * t*shs_coeffs[:,0] + 3. * mint * (t**2) * shs_coeffs[:, 1] + (t**3)*shs_coeffs[:, 2])
            shs[target_mask] += dshs.view(-1, 16, 3)
        else:
            shs = None

        
        # scales[target_mask] = scales[target_mask] * feat_exp # scale (N,3) temporal change function (N, 1)

        # start_time = time.time()
        # dx_coeffs_nn = smooth_feature_knn(dx_coeffs, rays_pts_emb,target_mask)
        # end_time = time.time()
        # elapsed_ms = (end_time - start_time) * 1000
        # print('knn ',elapsed_ms)
         
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

        return pts, rotations, opacity,shs, dx_coeffs
    
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

        self.deformation_net = Deformation(W=net_width,  args=args)

        self.apply(initialize_weights)

    def forward(self, point, rotations=None, scales=None, shs=None, times_sel=None, h_emb=None, iteration=None, target_mask=None):

        return  self.deformation_net(
            point,
            rotations,
            scales,
            shs,
            times_sel, 
            iteration=iteration, 
            h_emb=h_emb, 
            target_mask=target_mask
        )
        
    
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    
    def get_dyn_coefs(self, xyz):
        return self.deformation_net.get_dx_coeffs(xyz)

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