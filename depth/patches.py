import torch
import torch.nn.functional as F

def patchify(img, patch_size):
    # img: (H, W) → (1, 1, H, W) for unfolding
    img = img.unsqueeze(0).unsqueeze(0)
    patches = F.unfold(img, kernel_size=patch_size, stride=patch_size)  # (1, patch_area, num_patches)
    return patches.squeeze(0), img.shape[-2:]

def compute_patch_norms(patches):
    # patches: (patch_area, num_patches)
    norms = torch.norm(patches, dim=0, keepdim=True) + 1e-8  # avoid divide by zero
    return norms

def normalize_patches(patches, norms):
    return patches / norms

def reconstruct_image(patches, image_size, patch_size):
    # patches: (patch_area, num_patches)
    patches = patches.unsqueeze(0)  # (1, patch_area, num_patches)
    output = F.fold(patches, output_size=image_size, kernel_size=patch_size, stride=patch_size)
    
    # To normalize overlapping areas (if stride < patch_size, which we don't assume here)
    divisor = F.fold(torch.ones_like(patches), output_size=image_size, kernel_size=patch_size, stride=patch_size)
    return (output / divisor).squeeze()