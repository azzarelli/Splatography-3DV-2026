import torch
import cv2
import os
import matplotlib.pyplot as plt
root = f'/media/barry/56EA40DEEA40BBCD/DATA/dynerf/flame_salmon/cam01/images/'
image_fp = f'0000.png'

fp = os.path.join(root, image_fp)
bgr_image = cv2.imread(fp)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


filtered = cv2.bilateralFilter(rgb_image, d=9, sigmaColor=75, sigmaSpace=75)

plt.figure(figsize=(10, 5))

# Original (RGB)
plt.subplot(1, 2, 1)
plt.imshow(rgb_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Filtered (grayscale)
plt.subplot(1, 2, 2)
plt.imshow(filtered, cmap='gray')
plt.title("Bilateral Filtered (Grayscale)")
plt.axis('off')

plt.tight_layout()
plt.show()