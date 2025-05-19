import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from torchvision import transforms as T
from PIL import Image
import torch.nn.functional as F
import torch

# ========================== CONFIGURATION ==========================
root = '/media/barry/56EA40DEEA40BBCD/DATA/dynerf/flame_salmon'
W, H = 2704//2, 2028//2
focal_x = focal_y = 1458.4999683

patch_size = 4

# ========================== MAIN ==========================
# Load point clouds
frames = [f'{i:04}' for i in range(50)]
transform = T.ToTensor()

patchdata = {}
for cam_dir in ['cam01', 'cam10', 'cam11', 'cam20']:
    patchdata[cam_dir] = []
    output_fp = os.path.join(root, 'heatmaps', f'{cam_dir}.png')

    patch_std_over_time = []
    for index, frame in enumerate(frames):
        frame_fp = os.path.join(root, cam_dir, 'images', f'{frame}.png')
        img = Image.open(frame_fp)
        img = transform(img)
        
        # patches , N, 3, patch*patch
        patches = F.unfold(img.unsqueeze(0), kernel_size=patch_size, stride=patch_size)  # (1, C*patch*patch, N_patches)
        patches = patches.squeeze(0).transpose(0, 1)  # shape: (N_patches, patch_dim)
        
        patch_std = patches.std(dim=1)  # shape: (N_patches,)
        # patch_std_over_time.append(patch_std)
        patch_std_over_time.append(patch_std) 
    
    patch_std_over_time = torch.stack(patch_std_over_time, dim=0)
    std_variance_per_patch = patch_std_over_time.var(dim=0)  # shape: (N_patches,)
    grid_h = H // patch_size
    grid_w = W // patch_size
    heatmap = std_variance_per_patch.view(grid_h, grid_w)  # shape: (H//16, W//16)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_up = F.interpolate(
    heatmap_norm.unsqueeze(0).unsqueeze(0),  # (1, 1, grid_h, grid_w)
    size=(H, W),
    mode='bilinear',
    align_corners=False
    ).squeeze().numpy()  # Final shape: (H, W)

    # ========== PLOT ==========
    # Choose a few patches to visualize
    num_patches = patch_std_over_time.shape[1]
    # pids = [i for i in range(num_patches) if i % 50 == 0]
    # plt.figure(figsize=(10, 6))
    # for pid in pids:
    #     plt.plot(patch_std_over_time[:, pid].numpy(), label=f'Patch {pid}')
    # plt.xlabel('Frame')
    # plt.ylabel('Std Dev')
    # plt.title(f'Standard Deviation Over Time for Selected Patches ({cam_dir})')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # Plot the variance w.r.t patch number (x axis doesnt mean anything here)
    # std_variance_per_patch = patch_std_over_time.var(dim=0)  # shape: (N_patches,)
    # print(std_variance_per_patch.shape)
    # exit()
    # plt.figure(figsize=(10, 4))
    # plt.plot(std_variance_per_patch.numpy())
    # plt.xlabel('Patch Index')
    # plt.ylabel('Variance of Std over Time')
    # plt.title('Temporal Variance of Patch Std')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    threshold = 1e-03 
    
    masked_heatmap = np.ma.masked_less(heatmap_up, threshold)
    cmap = cm.inferno
    cmap.set_bad(color='gray')  # or 'black', 'white', etc.

    # plt.figure(figsize=(12, 6))
    # plt.imshow(masked_heatmap, cmap=cmap, vmin=0, vmax=1)
    # plt.colorbar(label='Normalized Variance of Patch Std over Time')
    # plt.title(f'Heatmap with Threshold ({threshold})')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    fixed_low_value = 0.1
    normalized_min = 0.2
    normalized_max = 1.0

    # Copy to avoid modifying original
    adjusted_heatmap = heatmap_up.copy()

    # Create masks
    low_mask = adjusted_heatmap < threshold
    high_mask = ~low_mask

    # Set all low values to fixed value
    adjusted_heatmap[low_mask] = fixed_low_value

    # Normalize high values to [0.2, 1.0]
    high_values = adjusted_heatmap[high_mask]
    if high_values.size > 0:
        # Rescale high values to [0.2, 1.0]
        high_values_rescaled = (high_values - high_values.min()) / (high_values.max() - high_values.min() + 1e-8)
        high_values_rescaled = high_values_rescaled * (normalized_max - normalized_min) + normalized_min
        adjusted_heatmap[high_mask] = high_values_rescaled
        
    gray_img = (adjusted_heatmap * 255).clip(0, 255).astype(np.uint8)

    # Repeat across RGB channels
    rgb_img = np.stack([gray_img]*3, axis=-1)  # shape: (H, W, 3)
    Image.fromarray(rgb_img).save(output_fp)
    
    # plt.figure(figsize=(12, 6))
    # plt.imshow(adjusted_heatmap, cmap='inferno', vmin=0, vmax=1)
    # plt.colorbar(label='Adjusted Variance Map')
    # plt.title('Adjusted Heatmap (0.1 for low, normalized high)')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
exit()