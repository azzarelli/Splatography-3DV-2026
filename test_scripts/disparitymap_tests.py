import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the left image and disparity map
left_image = cv2.imread("./data/disparity/images/left.png")  # Change to your image path
disparity_map = cv2.imread("./data/disparity/disparity/sceneflow.png", cv2.IMREAD_GRAYSCALE)
ground_truth_right = cv2.imread("./data/disparity/images/right.png")  # Ground truth right image

# Convert disparity to float
disparity_map = disparity_map.astype(np.float32)

mean_disp = np.mean(disparity_map)
min_disp = np.min(disparity_map)


# Get image dimensions
h, w, _ = left_image.shape

# Initialize an empty right image (same shape as left image)
estimated_right = np.zeros_like(left_image)

# Warp pixels from left to right image
for y in range(h):
    for x in range(w):
        d = int(disparity_map[y, x])  # Read disparity value
        x_new = x - (d)  # Compute new x position in right image

        if 0 <= x_new < w:
            estimated_right[y, x_new] = left_image[y, x]

# Fill gaps using inpainting (optional)
# mask = (estimated_right == 0).all(axis=2).astype(np.uint8) * 255
# estimated_right = cv2.inpaint(estimated_right, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Save or display the generated right image
# cv2.imwrite("right_image.png", right_image)
# ground_truth_right = cv2.cvtColor(ground_truth_right, cv2.COLOR_BGR2RGB)
# estimated_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

# ground_truth_right = cv2.cvtColor(ground_truth_right, cv2.COLOR_BGR2RGB)
# estimated_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

ground_truth_gray = cv2.cvtColor(ground_truth_right, cv2.COLOR_BGR2GRAY)
estimated_gray = cv2.cvtColor(estimated_right, cv2.COLOR_BGR2GRAY)

difference = cv2.absdiff(ground_truth_gray, estimated_gray)

# Display images and difference side by side
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(ground_truth_right, cv2.COLOR_BGR2RGB))
plt.title("Ground Truth Right Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(estimated_right, cv2.COLOR_BGR2RGB))
plt.title("Estimated Right Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(difference, cmap="hot")  # Heatmap for better visualization
plt.colorbar()
plt.title("ABS-Difference between GT and Pred")
plt.axis("off")

plt.show()
