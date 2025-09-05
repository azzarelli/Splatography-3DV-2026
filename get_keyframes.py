
"""
For each frame, we should calculate the L2 distance between the current and key frame list. We should sum this up for
all views and keyframes in the list and find the max distance and then append it to the keyframe list
"""
import os
import cv2
import numpy as np

root = "/media/barry/56EA40DEEA40BBCD/DATA/Condense/Curling/train"

cam_list = sorted(os.listdir(root))

cam_data={}
total_frames=0
for idx, cam_dir in enumerate(cam_list):
    cam_dir = os.path.join(root, cam_dir, 'depth_corrected')
    file_list = sorted(os.listdir(cam_dir)) # get each file
    file_data = {f"{i}":os.path.join(cam_dir, f) for i, f in enumerate(file_list) }
    cam_data[idx] = file_data
    total_frames = len(file_list)

def load_depth_img(frame_path):# shape 640, 576
    return cv2.imread(frame_path,cv2.IMREAD_UNCHANGED).astype(np.float32).flatten() / 1000.0 

   
# Initialise keyframe indexing
keyframes = [0, total_frames-1]
padding = 100 # apply if we want to avoid selecting new key frames that are too close to the current frames
N = 2
for n in range(N): # for N keyframes that we want to find
    # make pairs for each set of keyframes
    keyframes = sorted(keyframes)
    print(F"Updated keyframes: {keyframes}")

    n = len(keyframes)-1
    keyframe_pairs = []
    for i in range(n):
        keyframe_pairs.append((keyframes[i], keyframes[i+1]))
    print(keyframe_pairs)
    # Cycle through each pair and determine a new inbetween keyframes
    for min_frame, max_frame in keyframe_pairs:
        frame_scores = [0. for i in range(total_frames)]
        
        for cam in cam_data.keys(): # Loop through each camera/viewpoint
            # Load keyframes as vectors
            min_kf = load_depth_img(cam_data[cam][f"{min_frame}"])
            max_kf = load_depth_img(cam_data[cam][f"{max_frame}"])
            
            for i in range(min_frame+1+padding, max_frame-padding): # Loop all frames that lie between the first and next keyframe in the pair
                # Load and filter current frame
                frame = load_depth_img(cam_data[cam][f"{i}"])
                frame_mask = (frame > 0.3) & (frame < 3.5) # Analyse the data a bit to figure out what should be inbound
                    
                    
                frame_scores[i] += ((frame[frame_mask] - min_kf[frame_mask])**2).mean()
                frame_scores[i] += ((frame[frame_mask] - max_kf[frame_mask])**2).mean()
                    
        max_score = -0.1
        next_kf_idx = -1
        for i, s in enumerate(frame_scores):
            if s > max_score:
                next_kf_idx = i
                max_score = s
        print(next_kf_idx)
        padding = int(padding/2)
        keyframes.append(next_kf_idx)
end = sorted([float(kf/299) for kf in keyframes])
print(end)
    