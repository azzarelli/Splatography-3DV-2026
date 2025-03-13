import torch
import torch.nn as nn
import numpy as np
import utils.rotationcontinuity_tools as tools

class Model(nn.Module):
    def __init__(self, out_rotation_mode="Quaternion", regress_t=False):
        super(Model, self).__init__()
        
        self.out_rotation_mode = out_rotation_mode
        self.regress_t =regress_t
        
        if(out_rotation_mode == "Quaternion"):
            self.out_channel = 4
        elif (out_rotation_mode  == "ortho6d"):
            self.out_channel = 6
        elif (out_rotation_mode == "ortho5d"):
            self.out_channel = 5
        elif (out_rotation_mode  == "rmat"):
            self.out_channel = 9
        elif (out_rotation_mode == "axisAngle"):
            self.out_channel = 4
        elif (out_rotation_mode == "euler"):
            self.out_channel = 3
        
        if(regress_t==True):
            self.out_channel = self.out_channel+3
            
        #in b*point_num*3
        #out b*1*512
        self.feature_extracter = nn.Sequential(
                nn.Conv1d(3, 64, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv1d(128, 1024, kernel_size=1),
                nn.AdaptiveMaxPool1d(output_size=1)
                )
        
        #in b*1024
        #out b*out_channel
        self.mlp = nn.Sequential(
                nn.Linear(2048, 512),
                nn.LeakyReLU(),
                nn.Linear(512, self.out_channel))
        
    #pt b*point_num*3
    def forward(self, pt1, pt2):
        batch = pt1.shape[0]        
        # the input is batch, 3, n points, and the below functionality puts out num points in the batch positon
        # Im assumin this is correct as num points may refer to all points in the a group undergoing the same rotation mayhe? idk
        
        feature_pt1 = self.feature_extracter(pt1.unsqueeze(-1)).view(batch,-1)#b*512
        feature_pt2 = self.feature_extracter(pt2.unsqueeze(-1)).view(batch,-1)#b*512
        
        f = torch.cat((feature_pt1, feature_pt2), 1) #batch*1024
        
        out_data = self.mlp(f)#batch*out_channel
        if(self.regress_t == True):
            out_rotation = out_data[:,3:3+self.out_channel] #batch*(out_channel-3)
        else:
            out_rotation = out_data #batch*out_channel
        
        if(self.out_rotation_mode == "Quaternion"):
            out_r_mat = tools.compute_rotation_matrix_from_quaternion(out_rotation) #b*3*3
        elif(self.out_rotation_mode=="ortho6d"):
            out_r_mat = tools.compute_rotation_matrix_from_ortho6d(out_rotation) #b*3*3
        elif(self.out_rotation_mode=="ortho5d"):
            out_r_mat = tools.compute_rotation_matrix_from_ortho5d(out_rotation) #b*3*3
        elif ((self.out_rotation_mode=="rmat") and (self.training==True)):
            out_r_mat = out_rotation.view(batch,3,3)#b*3*3
        elif ((self.out_rotation_mode=="rmat") and (self.training==False)):
            out_r_mat = out_rotation.view(batch,3,3)#b*3*3
            out_r_mat = tools.compute_rotation_matrix_from_matrix(out_r_mat)
        elif(self.out_rotation_mode=="axisAngle"):
            out_r_mat = tools.compute_rotation_matrix_from_axisAngle(out_rotation)#b*3*3
        elif(self.out_rotation_mode=="euler"):
            out_r_mat = tools.compute_rotation_matrix_from_euler(out_rotation)#b*3*3
                    
        return out_r_mat

        
        
        
        
        
        
        
    