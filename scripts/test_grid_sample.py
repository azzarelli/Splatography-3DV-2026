import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def test_grid_sample_rgb(C=3, H=5, W=5, align_corners=True):
    # Step 1: Create a synthetic RGB grid [1, 3, H, W]
    grid = torch.zeros(1, C, H, W)

    # Create a diagonal gradient from top-left (0) to bottom-right (1)
    y = torch.linspace(0, 1, H).view(H, 1)  # Vertical
    x = torch.linspace(0, 1, W).view(1, W)  # Horizontal
    diag = (x + y) / 2  # Combine into diagonal gradient, shape [H, W]

    # Set RGB channels (you can vary how each channel uses the gradient)
    grid[0, 0] = diag            # Red channel = diagonal gradient
    grid[0, 1] = diag.pow(0.5)   # Green = brighter diagonal (sqrt)
    grid[0, 2] = 1 - diag        # Blue = inverse diagonal

    # Step 2: Generate sampling coordinates at cell centers
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    xv, yv = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([xv, yv], dim=-1).reshape(-1, 2)  # [N, 2]
    coords = coords.unsqueeze(0)  # Add batch dim [1, N, 2]
    print(coords.shape)
    # Step 3: Sample from the grid
    # coords[..., 0] = 0.
    print(grid.shape)
    sampled = F.grid_sample(grid, coords.view(1, 1, -1, 2), align_corners=align_corners)
    sampled = sampled.view(3, -1).transpose(0, 1)  # [N, 3]

    print(sampled.shape)

    # Step 4: Visualization
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # a) Original grid
    axs[0].imshow(grid.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original Grid")
    axs[0].axis("off")

    # b) Sampled colors (plotted as patches)
    axs[1].imshow(torch.ones(H, W, 3))  # White background

    # Convert normalized coords to pixel space
    if align_corners:
        x_px = ((coords[0, :, 0] + 1) / 2) * (W - 1)
        y_px = ((coords[0, :, 1] + 1) / 2) * (H - 1)
    else:
        x_px = ((coords[0, :, 0] + 1) * W - 1) / 2
        y_px = ((coords[0, :, 1] + 1) * H - 1) / 2
    x_px = x_px.long()
    y_px = y_px.long()
    
    g = torch.zeros_like(grid.squeeze())
    
    g[:,y_px, x_px] = sampled.permute(1,0)
    
    axs[1].imshow(g.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[1].set_title("Original Grid")
    axs[1].axis("off")

    # c) Overlay sample points on original
    axs[2].imshow(grid.squeeze().permute(1, 2, 0).cpu().numpy())
    x_px = ((coords[0, :, 0] + 1) * (W - 1) / 2).numpy()
    y_px = ((coords[0, :, 1] + 1) * (H - 1) / 2).numpy()
    axs[2].scatter(x_px, y_px, c='white', edgecolors='black')
    axs[2].set_title("Sample Points on Grid")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

test_grid_sample_rgb(H=25, W=3, align_corners=True)