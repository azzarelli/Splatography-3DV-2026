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
lpips_vgg = lpips.LPIPS(net='vgg').to(device)
lpips_alex = lpips.LPIPS(net='alex').to(device)

# Paths
root = "/home/barry/Desktop/PhD/SparseViewPaper/DYNERF_RESULTS"
gt_root = os.path.join(root, "GT")

scenes = ["CSP", "FST", "FSM"]
scenes = ["FST"]

# models = ["4DGS", "W4DGS"]
models = ["4DGS"]

# Frame-wise results dictionary


def preprocess(img):
    """Prepares an image for LPIPS (PyTorch tensor, GPU, scaled to [-1, 1])"""
    tensor = torch.from_numpy(img).float().permute(2, 0, 1).to(device)
    return tensor

with torch.no_grad():
    finaldictionary = {}
    for scene in scenes:
        if scene not in finaldictionary.keys():
            finaldictionary[scene]={
                m:{} for m in models
            }
            
        # Fetch and sort GT filepaths
        gt_dir = os.path.join(gt_root, scene)
        gt_img_fp = os.listdir(gt_dir)
        gt_img_fp = [os.path.join(gt_dir, f) for f in natsorted(gt_img_fp)]
        
        for model in models:
            cnt = 0
            per_frame_results = {i: {'full':{'mae':0.,'psnr': 0., 'ssim': 0., 
                                            'lpips_vgg': 0., 'lpips_alex': 0.},
                                    'mask':{'mae':0.,'psnr': 0.,'m_mae':0.,'m_psnr': 0., 'ssim': 0.,
                                    'lpips_vgg': 0., 'lpips_alex': 0.}} for i in range(50)}
            
            # Fetch and sort model test image filepaths
            test_fp = os.path.join(root, model, scene)
            test_img_fp = os.listdir(test_fp)
            test_img_fp = [os.path.join(test_fp, f) for f in natsorted(test_img_fp)]
            
            for test_fp, gt_fp in tqdm(zip(test_img_fp, gt_img_fp), total=50, desc=f"{scene} : {model}"):

                # Load and normalize images
                test_img = cv2.imread(test_fp, cv2.IMREAD_UNCHANGED) / 255.0
                gt_img_full = cv2.imread(gt_fp, cv2.IMREAD_UNCHANGED) / 255.0

                import matplotlib.pyplot as plt
                # Extract
                gt_img = gt_img_full[:, :, :3]
                mask = np.expand_dims( gt_img_full[:, :, -1], axis=-1)
                
                test_mask = test_img * mask
                gt_mask = gt_img * mask
                
                
                # plt.figure(figsize=(10, 5))

                # plt.subplot(1, 2, 1)
                # plt.imshow(test_mask)
                # plt.title("Test Image")
                # plt.axis("off")

                # plt.subplot(1, 2, 2)
                # plt.imshow(gt_mask)
                # plt.title("Ground Truth Image")
                # plt.axis("off")

                # plt.tight_layout()
                # plt.show()
                # exit()
                # LPIPS (GPU, no gradient tracking)
                gt_tensor = preprocess(gt_img)
                test_tensor = preprocess(test_img)
                mask_tensor = (preprocess(mask) > 0).long()

                # print(mask_tensor.shape, test_tensor.shape)
                # Full Reference
                per_frame_results[cnt]['full']['mae']  += np.abs(gt_img - test_img).mean()
                per_frame_results[cnt]['full']['psnr']  += psnr(gt_tensor, test_tensor).item()
                per_frame_results[cnt]['full']['ssim']  += ssim(gt_img, test_img, win_size=3, channel_axis=-1, data_range=1.0)
                # per_frame_results[cnt]['full']['lpips_vgg'] += lpips_vgg(gt_tensor, test_tensor).item()
                # per_frame_results[cnt]['full']['lpips_alex'] += lpips_alex(gt_tensor, test_tensor).item()
                
                # # Masked
                per_frame_results[cnt]['mask']['mae']  += (gt_tensor*mask_tensor - test_tensor*mask_tensor).abs().mean().item()
                per_frame_results[cnt]['mask']['m_mae']  += ((gt_tensor*mask_tensor - test_tensor*mask_tensor).abs().sum() / mask_tensor.sum().float()).item()
                per_frame_results[cnt]['mask']['psnr']  += psnr(gt_tensor*mask_tensor, test_tensor*mask_tensor).item()
                per_frame_results[cnt]['mask']['m_psnr']  += psnr(gt_tensor, test_tensor, mask_tensor.squeeze(0)).item()
                per_frame_results[cnt]['mask']['ssim']  += ssim(gt_mask, test_mask, win_size=3, channel_axis=-1, data_range=1.0)
                print(per_frame_results[cnt])
                # per_frame_results[cnt]['mask']['lpips_vgg'] += lpips_vgg(gt_tensor*mask_tensor, test_tensor*mask_tensor).item()
                # per_frame_results[cnt]['mask']['lpips_alex'] += lpips_alex(gt_tensor*mask_tensor, test_tensor*mask_tensor).item()

                # Frame counter (reset every 300 frames)
                cnt = cnt + 1 

            averages={'full':{
                'mae':0.,'psnr': 0., 'ssim': 0., 'lpips_vgg': 0., 'lpips_alex': 0.},'mask':{'mae':0.,'psnr': 0.,'m_mae':0.,'m_psnr': 0., 'ssim': 0., 'lpips_vgg': 0., 'lpips_alex': 0.}
            }
            # Sum up all per frame results
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
            
            # Get the average
            averages['full']['mae'] /= 50
            averages['full']['psnr'] /= 50
            averages['full']['ssim'] /= 50
            averages['full']['lpips_vgg'] /= 50
            averages['full']['lpips_alex'] /= 50

            averages['mask']['mae'] /= 50
            averages['mask']['m_mae'] /= 50
            averages['mask']['psnr'] /= 50
            averages['mask']['m_psnr'] /= 50
            averages['mask']['ssim'] /= 50
            averages['mask']['lpips_vgg'] /= 50
            averages['mask']['lpips_alex'] /= 50

            finaldictionary[scene][model] = {
                "average":averages,
                "per-frame": per_frame_results
            }
import json
with open("debugging/resulta.json", 'w') as json_file:
    json.dump(finaldictionary, json_file,  indent=4)