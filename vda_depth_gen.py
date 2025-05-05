import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ========================== CONFIGURATION ==========================
root = '/media/barry/56EA40DEEA40BBCD/DATA/dynerf/flame_salmon'
W, H = 2704, 2028
focal_x = focal_y = 1458.4999683

# ========================== INTRINSICS ==========================
def project_points(points, width, height, fx, fy):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Filter out invalid points (z <= 0)
    valid = z > 1e-6
    x, y, z = x[valid], y[valid], z[valid]

    cx, cy = (width - 1) / 2, (height - 1) / 2
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    uv = np.zeros((points.shape[0], 2))  # initialize to zeros
    uv[valid, 0] = u
    uv[valid, 1] = v
    return uv

# ========================== MASK LOADING ==========================
def load_mask_from_png(path, erode_kernel_size=2):
    mask_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask_img is None or mask_img.shape[2] < 4:
        raise ValueError(f"Invalid mask image at {path}")

    alpha = mask_img[:, :, 3] > 0
    alpha = alpha.astype(np.uint8)
    # Erode the mask
    if erode_kernel_size > 0:
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        alpha = cv2.erode(alpha, kernel, iterations=1)

    return alpha.astype(bool)

# ========================== MASK DEPTH ==========================
def create_depth_image(points, width, height, fx, fy):
    uv = project_points(points, width, height, fx, fy)
    u = np.floor(uv[:, 0]).astype(int)
    v = np.floor(uv[:, 1]).astype(int)
    z = points[:, 2]

    # Filter valid pixel indices and depths
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z > 0)
    u, v, z = u[valid], v[valid], z[valid]

    # Flatten 2D indices for use in 1D arrays
    indices = v * width + u

    # Use np.unique to keep the closest depth per pixel
    unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
    min_depths = np.full(unique_indices.shape, np.inf)

    # Choose minimum depth for each pixel
    np.minimum.at(min_depths, inverse_indices, z)

    # Reconstruct depth image
    depth_image = np.zeros((height * width,), dtype=np.float32)
    depth_image[unique_indices] = min_depths
    return depth_image.reshape((height, width))

# ========================== MAIN ==========================
# Load point clouds
frames = [f'{i:04}' for i in range(50)]

for cam_dir in ['cam01', 'cam10', 'cam11', 'cam20']:
    pcd0 = o3d.io.read_point_cloud(f"{root}/vda-clouds/{cam_dir}/point0000.ply")
    points0 = np.asarray(pcd0.points)
    depth0 = create_depth_image(points0, W, H, focal_x, focal_y)

    mask0 = load_mask_from_png(f'{root}/static_masks/{cam_dir}.png')
    mask0_resized = cv2.resize(mask0.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Only consider valid depth values inside the mask
    person_depth_values = depth0[(depth0 > 0) & mask0_resized]
    depth_min, depth_max = np.percentile(person_depth_values, [2, 98])  # Robust estimate
    depth_min -= 0.05  # meters tolerance
    depth_max += 0.05  # meters tolerance
    print(f"Target depth range: {depth_min:.2f} to {depth_max:.2f} with margin {0.1}")
    
    output_dir = f"./filtered_depth_frames"
    os.makedirs(output_dir, exist_ok=True)
    video_out = cv2.VideoWriter(f"./depth_video_gray.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H), isColor=False)


    for index, frame in enumerate(frames):
        if index > 0 :
            print(index)

            pcd1 = o3d.io.read_point_cloud(f"{root}/vda-clouds/{cam_dir}/point{frame}.ply")

            # Load and mask points
            points1 = np.asarray(pcd1.points)
            colors1 = np.asarray(pcd1.colors)

            depth = create_depth_image(points1, W, H, focal_x, focal_y)
            depth_filtered = np.where((depth >= depth_min) & (depth <= depth_max), depth, 0)

            normalized = np.zeros_like(depth_filtered, dtype=np.uint8)
            valid = depth_filtered > 0
            if np.any(valid):
                normalized[valid] = ((depth_filtered[valid] - depth_min) /
                                    (depth_max - depth_min) * 255).astype(np.uint8)

            # Save to video
            video_out.write(normalized)
    video_out.release()
    exit()
                
        #     plt.figure(figsize=(10, 8))
        #     plt.imshow(depth, cmap='gray', origin='upper')
        #     plt.axis('off')
        #     plt.show()
        # if index > 2:
        #     exit()                          

    # # Normalize based only on valid depth values inside the mask
    # # First resize depth to match the mask size
    # norm_depth_resized = cv2.resize(depth_image, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)

    # # Compute normalization only where mask is True and depth > 0
    # valid_mask = (norm_depth_resized > 0) & mask1
    # masked_depths = norm_depth_resized[valid_mask]

    # min_depth = masked_depths.min()
    # max_depth = masked_depths.max()

    # # Normalize only the masked valid values
    # norm_depth = np.zeros_like(norm_depth_resized, dtype=np.float32)
    # norm_depth[valid_mask] = (norm_depth_resized[valid_mask] - min_depth) / (max_depth - min_depth)

    # # Convert to 8-bit and save
    # depth_8bit = (norm_depth * 255).astype(np.uint8)
    # destination = f'{root}/vda/{cam_dir}/target_depth.png'
    # cv2.imwrite(destination, depth_8bit)
exit()


# ========================== DISPLAY DEPTH IMAGE ==========================

plt.figure(figsize=(10, 8))
plt.imshow(depth_display, cmap='gray', origin='upper')
plt.axis('off')
plt.show()

# ========================== DISPLAY DEPTH HISTOGRAM ==========================
plt.hist(depth_image[depth_image > 0].ravel(), bins=100)
plt.title("Depth Histogram")
plt.xlabel("Depth (z)")
plt.ylabel("Pixel Count")
plt.show()

# ========================== DISPLAY PCD RGB PROJECTION ==========================
def create_masked_rgb_image(points, colors, width, height, fx, fy):
    uv = project_points(points, width, height, fx, fy)
    u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(points.shape[0]):
        if 0 <= u[i] < width and 0 <= v[i] < height:
            rgb_image[v[i], u[i]] = colors[i]  # assuming colors are in [0,1]

    return rgb_image


masked_rgb_image = create_masked_rgb_image(masked_xyz1, masked_colors1, W, H, focal_x, focal_y)

plt.figure(figsize=(10, 8))
plt.imshow(masked_rgb_image)
plt.axis('off')
plt.title("Masked RGB Image", color='white')

plt.show()