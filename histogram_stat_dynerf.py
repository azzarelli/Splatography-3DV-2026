from natsort import natsorted
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr  # Assuming you're using your custom PSNR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

# Paths
root = "/home/barry/Desktop/PhD/SparseViewPaper/DYNERF_RESULTS"
gt_root = os.path.join(root, "GT")

scenes = [ "FST", "FSM"] #"CSP",
scenes = ["FSM"]

models = ["4DGS", "W4DGS", "Ours"]
# models = ["Ours"]

# Frame-wise results dictionary
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
            per_frame_results = {i: 0. for i in range(50)}
            avg_res = 0.
            # Fetch and sort model test image filepaths
            test_fp = os.path.join(root, model, scene)
            test_img_fp = os.listdir(test_fp)
            test_img_fp = [os.path.join(test_fp, f) for f in natsorted(test_img_fp)]
            for test_fp, gt_fp in tqdm(zip(test_img_fp, gt_img_fp), total=50, desc=f"{scene} : {model}"):
                # print(gt_fp,test_fp)                
                # Load and normalize images
                frame1 = cv2.imread(gt_fp, cv2.IMREAD_UNCHANGED)     # ground truth (BGRA)
                frame2 = cv2.imread(test_fp, cv2.IMREAD_UNCHANGED)   # test (BGRA or BGR)
                
                # Ensure frame1 has 4 channels (BGRA)
                if frame1.shape[2] != 4:
                    raise ValueError("frame1 must have 4 channels (BGRA)")

                # Extract RGB and Alpha from frame1
                alpha = frame1[:, :, 3].astype(np.float32) #/ 255.0  # Normalize alpha to [0,1]
                rgb1 = frame1[:, :, :3].astype(np.float32)/255.0

                # Make sure frame2 is also 3-channel RGB
                if frame2.shape[2] == 4:
                    rgb2 = frame2[:, :, :3].astype(np.float32)/255.0
                elif frame2.shape[2] == 3:
                    rgb2 = frame2.astype(np.float32)/255.0

                # Expand alpha to shape (H, W, 3) for broadcasting
                alpha_3ch = np.stack([alpha]*3, axis=2)

                # Multiply alpha mask with both images
                # frame1 = (frame1 * alpha_3ch).astype(np.uint8)
                # frame2 = (frame2 * alpha_3ch).astype(np.uint8)
                # print(frame2.shape)
                frame3 = np.abs(rgb1 - rgb2).mean(axis=-1)*alpha
                # plt.figure(figsize=(10, 5))

                # plt.subplot(1, 3, 1)
                # plt.imshow(rgb1)

                # plt.subplot(1, 3, 2)
                # plt.imshow(rgb2)
                # plt.subplot(1, 3, 3)
                # plt.imshow(frame3)

                # plt.tight_layout()
                # plt.show()
                # break
                exit()
                
                # Split into R, G, B channels
                # channels1 = cv2.split(frame3)
                channels2 = cv2.split(frame3)
                # channels2 = cv2.split(frame2)

                total_intersection = 0
                for ch1, ch2 in zip(channels1, channels2):
                    # Compute histograms
                    hist1 = cv2.calcHist([ch1], [0], None, [256], [0, 256])
                    # hist2 = cv2.calcHist([ch2], [0], None, [256], [0, 256])

                    # Normalize histograms
                    hist1 = cv2.normalize(hist1, hist1).flatten()#[1:]
                    # hist2 = cv2.normalize(hist2, hist2).flatten()[1:]

                    # import matplotlib.pyplot as plt

                    # plt.plot(hist1, label='Frame1 Channel')
                    # # plt.plot(hist2, label='Frame2 Channel')
                    # plt.title("Histogram Comparison")
                    # plt.xlabel("Pixel Intensity (0–255)")
                    # plt.ylabel("Normalized Frequency")
                    # plt.legend()
                    # plt.grid(True)
                    # plt.show()
                    # exit()
                    # Compute intersection
                    intersection = np.sum(hist1)
                    total_intersection += intersection#/(frame3.shape[0]*frame3.shape[1])
                # break
                # Average intersection over 3 channels (or just use the sum)
                avg_intersection = total_intersection / 3.0
                per_frame_results[cnt] = avg_intersection       
                avg_res += avg_intersection
                cnt = cnt + 1 
            averages = avg_res / 50.
            print(averages)
            finaldictionary[scene][model] = {
                "average":averages,
                "per-frame": per_frame_results
            }
import json
with open("debugging/histogram_intersection.json", 'w') as json_file:
    json.dump(finaldictionary, json_file,  indent=4)