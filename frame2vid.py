# ffmpeg misbehaving so using python
import cv2
import os
import glob
from tqdm import tqdm
from natsort import natsorted
import numpy as np
# === Configuration ===
FOLDER="cool/full"
SAVEDIR="Piano"
OUTPUT="cool_render"
DATSET="Condense"

# 
# exp_names = ["fg_loss", "no_bgloss", "unifiedH"]
exp_names = [ "unifieddyn4_nostaticdupe"]

fps = 30
# === Helper to crop image to dimensions divisible by 8 ===
def crop_to_divisible_by_8(img):
    h, w = img.shape[:2]
    new_h = h - (h % 8)
    new_w = w - (w % 8)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    return img[start_y:start_y+new_h, start_x:start_x+new_w]

def process_video(input_folder, output_video, fps):
    # === Gather and sort image files ===
    image_files = natsorted(glob.glob(os.path.join(input_folder, "*.png")))
    # print(image_files)
    # exit()
    # === Read first image to get frame size ===
    first_img = cv2.imread(image_files[0])
    first_img = crop_to_divisible_by_8(first_img)
    height, width = first_img.shape[:2]

    # === Setup VideoWriter ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' or 'avc1' for mp4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # === Process and write frames ===
    i = 0
    for file in tqdm(image_files, desc="Creating video"):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)  # Load with alpha
        if img is None:
            continue  # skip broken images

        img = crop_to_divisible_by_8(img)

        try:
            b, g, r, a = cv2.split(img)
            alpha = a.astype(float) / 255.0
            alpha = cv2.merge([alpha, alpha, alpha])  # 3 channel alpha
            rgb = cv2.merge([b, g, r]).astype(float)
            black = np.zeros_like(rgb)
            blended = (rgb * alpha + black * (1 - alpha)).astype(np.uint8)
        except:
            b, g, r = cv2.split(img)
            blended = cv2.merge([b, g, r]).astype(np.uint8)

        blended = cv2.resize(blended, (width, height))
        video_writer.write(blended)
        # if i == 300:break
        # i+=1
    video_writer.release()
    print("Video saved to:", output_video)


for exp in exp_names:
    input_folder=f"output/{DATSET}/{SAVEDIR}/{exp}/{FOLDER}"
    output_video=f"output/{DATSET}/{SAVEDIR}/{exp}/test_render.mp4"
    process_video(input_folder, output_video, fps)