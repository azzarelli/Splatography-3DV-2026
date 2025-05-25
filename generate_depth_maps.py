from submodules.DAV2.depth_anything_v2.dpt import DepthAnythingV2
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
# Initialize RGB to Depth model (DepthAnything v2)
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb', 'vitg'
depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 1.})
depth_model.load_state_dict(torch.load(f'submodules/DAV2/checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.cuda().eval()

fp_root = '/media/barry/56EA40DEEA40BBCD/DATA/Condense/'
scene = 'Bassist/'

root = f'{fp_root}{scene}train/'

cam_list = os.listdir(root)
for cam_name in cam_list:
    cam_root = f'{fp_root}{scene}train/{cam_name}/color_corrected'
    destination = f'{fp_root}{scene}train/{cam_name}/depth_mono'

    if not os.path.exists(destination):
        os.mkdir(destination)

    for f in tqdm.tqdm(os.listdir(cam_root)):
        img_fp = os.path.join(cam_root, f)
        img_dest = os.path.join(destination, f)
        img = Image.open(img_fp)
        img = np.array(img) / 255.    

        gt_depth =  depth_model.infer_image(img).cpu().numpy()
        # depth_normalized = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
        depth_8bit = (gt_depth * 255).astype('uint8')

        Image.fromarray(depth_8bit).save(img_dest)
        # print(gt_depth.shape)
        
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(gt_depth, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # exit()

        # gt_depth =  depth_model.infer_image(viewpoint_cam.original_image.permute(1,2,0).numpy())
