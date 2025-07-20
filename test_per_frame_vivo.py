from natsort import natsorted
import os
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr  # Assuming you're using your custom PSNR
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
result_dir = "/home/barry/Desktop/other_code/GS_Research/output/Condense/Pony/Test/images"
gt_dir = "/media/barry/56EA40DEEA40BBCD/DATA/Condense/Piano/test"

order = ["000809414712", "000875114712", "000906614712", "000950714712"]
image_folder = "scene_masks"

# File preparation
result_fp = os.listdir(result_dir)
test_fps = [os.path.join(result_dir, f) for f in natsorted(result_fp)]

gt_fps = []
for o in order:
    gt_fps += [os.path.join(gt_dir, o, image_folder, f) for f in natsorted(os.listdir(os.path.join(gt_dir, o, image_folder)))]


# Initialize LPIPS models on GPU
lpips_vgg = lpips.LPIPS(net='vgg').to(device)
lpips_alex = lpips.LPIPS(net='alex').to(device)


# Frame-wise results dictionary
cnt = 0
per_frame_results = {i: {'full':{'mae':0.,'psnr': 0., 'ssim': 0., 'lpips_vgg': 0., 'lpips_alex': 0.},'mask':{'mae':0.,'psnr': 0.,'m_mae':0.,'m_psnr': 0., 'ssim': 0., 'lpips_vgg': 0., 'lpips_alex': 0.}} for i in range(1200)}

def preprocess(img):
    """Prepares an image for LPIPS (PyTorch tensor, GPU, scaled to [-1, 1])"""
    tensor = torch.from_numpy(img).float().permute(2, 0, 1).to(device)
    return tensor

with torch.no_grad():
    for test_fp, gt_fp in tqdm(zip(test_fps, gt_fps), total=len(test_fps), desc="Processing frames"):
        # Load and normalize images
        test_img = cv2.imread(test_fp, cv2.IMREAD_UNCHANGED) / 255.0
        gt_img_full = cv2.imread(gt_fp, cv2.IMREAD_UNCHANGED) / 255.0
        
        # Extract
        gt_img = gt_img_full[:, :, :3]
        mask = np.expand_dims( gt_img_full[:, :, -1], axis=-1)
        
        test_mask = test_img * mask
        gt_mask = gt_img * mask

        # LPIPS (GPU, no gradient tracking)
        gt_tensor = preprocess(gt_img)
        test_tensor = preprocess(test_img)
        mask_tensor = (preprocess(mask) > 0).long()

        # print(mask_tensor.shape, test_tensor.shape)
        # Full Reference
        per_frame_results[cnt]['full']['mae']  += np.abs(gt_img - test_img).mean()
        per_frame_results[cnt]['full']['psnr']  += psnr(gt_tensor, test_tensor).item()
        per_frame_results[cnt]['full']['ssim']  += ssim(gt_img, test_img, win_size=3, channel_axis=-1, data_range=1.0)
        per_frame_results[cnt]['full']['lpips_vgg'] += lpips_vgg(gt_tensor, test_tensor).item()
        per_frame_results[cnt]['full']['lpips_alex'] += lpips_alex(gt_tensor, test_tensor).item()
        
        # Masked
        per_frame_results[cnt]['mask']['mae']  += (gt_tensor*mask_tensor - test_tensor*mask_tensor).abs().mean().item()
        per_frame_results[cnt]['mask']['m_mae']  += ((gt_tensor*mask_tensor - test_tensor*mask_tensor).abs().sum() / mask_tensor.sum().float()).item()
        per_frame_results[cnt]['mask']['psnr']  += psnr(gt_tensor*mask_tensor, test_tensor*mask_tensor).item()
        per_frame_results[cnt]['mask']['m_psnr']  += psnr(gt_tensor, test_tensor, mask_tensor.squeeze(0)).item()
        per_frame_results[cnt]['mask']['ssim']  += ssim(gt_mask, test_mask, win_size=3, channel_axis=-1, data_range=1.0)
        
        per_frame_results[cnt]['mask']['lpips_vgg'] += lpips_vgg(gt_tensor*mask_tensor, test_tensor*mask_tensor).item()
        per_frame_results[cnt]['mask']['lpips_alex'] += lpips_alex(gt_tensor*mask_tensor, test_tensor*mask_tensor).item()

        # Frame counter (reset every 300 frames)
        cnt = cnt + 1 #if cnt < 299 else 0

# Average results over the 4 sequences
for key in per_frame_results.keys():
    per_frame_results[key]['full']['mae'] /= 4
    per_frame_results[key]['full']['psnr'] /= 4
    per_frame_results[key]['full']['ssim'] /= 4
    per_frame_results[key]['full']['lpips_vgg'] /= 4
    per_frame_results[key]['full']['lpips_alex'] /= 4
    
    per_frame_results[key]['mask']['mae'] /= 4
    per_frame_results[key]['mask']['m_mae'] /= 4
    per_frame_results[key]['mask']['psnr'] /= 4
    per_frame_results[key]['mask']['m_psnr'] /= 4
    per_frame_results[key]['mask']['ssim'] /= 4
    per_frame_results[key]['mask']['lpips_vgg'] /= 4
    per_frame_results[key]['mask']['lpips_alex'] /= 4

averages={'full':{
    'mae':0.,'psnr': 0., 'ssim': 0., 'lpips_vgg': 0., 'lpips_alex': 0.},'mask':{'mae':0.,'psnr': 0.,'m_mae':0.,'m_psnr': 0., 'ssim': 0., 'lpips_vgg': 0., 'lpips_alex': 0.}
}

for key in per_frame_results.keys():
    averages['full']['mae'] += per_frame_results[key]['full']['mae']
    averages['full']['psnr'] += per_frame_results[key]['full']['psnr']
    averages['full']['ssim'] += per_frame_results[key]['full']['ssim']
    averages['full']['lpips_vgg'] += per_frame_results[key]['full']['lpips_vgg']
    averages['full']['lpips_alex'] += per_frame_results[key]['full']['lpips_alex']
    
    averages['mask']['mae'] += per_frame_results[key]['mask']['mae']
    averages['mask']['m_mae'] += per_frame_results[key]['mask']['m_mae']
    averages['mask']['psnr'] += per_frame_results[key]['mask']['psnr']
    averages['mask']['m_psnr'] += per_frame_results[key]['mask']['m_psnr']
    averages['mask']['ssim'] += per_frame_results[key]['mask']['ssim']
    averages['mask']['lpips_vgg'] += per_frame_results[key]['mask']['lpips_vgg']
    averages['mask']['lpips_alex'] += per_frame_results[key]['mask']['lpips_alex']

averages['full']['mae'] /= 4
averages['full']['psnr'] /= 4
averages['full']['ssim'] /= 4
averages['full']['lpips_vgg'] /= 4
averages['full']['lpips_alex'] /= 4

averages['mask']['mae'] /= 4
averages['mask']['m_mae'] /= 4
averages['mask']['psnr'] /= 4
averages['mask']['m_psnr'] /= 4
averages['mask']['ssim'] /= 4
averages['mask']['lpips_vgg'] /= 4
averages['mask']['lpips_alex'] /= 4

final_results = {
    "average":averages,
    "per-frame": per_frame_results
}
import json
with open("./pinaist_results.json", 'w') as json_file:
    json.dump(final_results, json_file,  indent=4)