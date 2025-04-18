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
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 4))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 16*3))

        # Same for opacity grid
      
        # self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_h = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_w = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.opacity_mu = nn.Sequential(nn.ReLU(),nn.Linear(net_size,net_size),nn.ReLU(),nn.Linear(net_size, 1))


    def query_time(self, rays_pts_emb, time_emb, iterations, mask=None):
        # Sample target points
        sp, st = self.grid(rays_pts_emb[mask,:3], time_emb[mask,:1],sep=True)
        sp_features = self.space_enc(sp)
        st_features = self.spacetime_enc(st)
        return sp_features, st_features

    def forward(self,rays_pts_emb, rotations_emb, shs_emb, time_emb, iteration, h_emb, target_mask):
        
        sp_features, st_features = self.query_time(rays_pts_emb, time_emb, iteration,target_mask)
        
        pts = rays_pts_emb + 0. #.clone()
        
        # 3rd order Bezier deformation for target points
        dx_coeffs = self.pos_coeffs(sp_features).view(-1, 3, 3) # N, 3, 3 
        t = time_emb[0:1]
        mint = 1. - t
        pts[target_mask] += 3. * (mint**2) * t*dx_coeffs[:,0] + 3. * mint * (t**2) * dx_coeffs[:, 1] + (t**3) *  dx_coeffs[:, 2]

        # Change in rotation
        rotations = rotations_emb + 0.
        rotations[target_mask] += self.rotations_deform(st_features)        
        
        if shs_emb is not None:
            shs = shs_emb + 0.
            dshs = self.shs_deform(st_features).view(-1, 16, 3)
            shs[target_mask] += dshs
        else:
            shs = None

        if h_emb is not None:
            opacity = h_emb + 0.
            
            w = (self.opacity_w(sp_features)**2)
            mu = torch.sigmoid(self.opacity_mu(sp_features))
            
            feat_exp = torch.exp(-w * (t-mu)**2)
            
            opacity[target_mask]  = h_emb[target_mask] * feat_exp
        else:
            opacity = None
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

def smooth_feature_knn(feature_masked, rays_pts_emb, target_mask, k=3):
    with torch.no_grad():
        coords = rays_pts_emb[target_mask, :3]  # (N_masked, 3)
        N_masked = coords.size(0)
        feat_shape = feature_masked.shape[1:]
        feature_dim = feature_masked.view(N_masked, -1).shape[1]

        edge_index = knn_graph(coords, k=k, loop=True)  # (2, N_masked * k)
        row, col = edge_index  # row[i] is center, col[i] is neighbor

        # Flatten features for accumulation
        feature_flat = feature_masked.view(N_masked, -1)  # (N_masked, F)
        neighbor_feat = feature_flat[col]                 # (E, F)

        # Accumulate features for each row index
        summed = torch.zeros_like(feature_flat)  # (N_masked, F)
        counts = torch.zeros(N_masked, 1, device=feature_masked.device)

        summed.index_add_(0, row, neighbor_feat)
        counts.index_add_(0, row, torch.ones_like(row, dtype=torch.float32).unsqueeze(1).to(feature_masked.device))

        smoothed_flat = summed / counts.clamp(min=1.0)
        return smoothed_flat.view(N_masked, *feat_shape)
    
    
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width

        self.deformation_net = Deformation(W=net_width,  args=args)

        self.apply(initialize_weights)

    def forward(self, point, rotations=None, shs=None, times_sel=None, h_emb=None, iteration=None, target_mask=None):

        return  self.deformation_net(
            point,
            rotations,
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