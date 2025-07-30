import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights).to(device).eval()

resize = T.Resize((1280, 720))  # << Reduce image size to avoid OOM

def load_rgba_image(path, is_gt=False):
    img_ = Image.open(path).convert("RGBA")
    if is_gt: alpha = TF.to_tensor(img_)[-1:] 
    img = TF.to_tensor(img_)[:3] * alpha  if is_gt else TF.to_tensor(img_)[:3] # Remove alpha
    
    if img.shape[1] == 2560:
        A=59
        B=2500
        C=34
        D=1405
        img = img[:, A:B,C:D]
        if is_gt: alpha = alpha[:, A:B,C:D]
    img = resize(img)    
    if is_gt:
        alpha = resize(alpha)    
        return img,alpha

    return img

def compute_raft_flow(img1, img2, model, device):
    img1 = img1.to(device).unsqueeze(0)
    img2 = img2.to(device).unsqueeze(0)
    with torch.no_grad():
        flow = model(img1, img2)[-1]
    return flow.squeeze().cpu().numpy()  # shape: [2, H, W]

def mse_between_flows(flow1, flow2):
    return np.mean((flow1 - flow2) ** 2)


def optical_flow_mse(test_folder, gt_folder, model):
    gt_files = sorted(os.listdir(gt_folder), key=lambda f: int(f.split('.')[0]))
    test_files = sorted(os.listdir(test_folder), key=lambda f: int(f.split('.')[0]))

    mses = []
    progress_bar = tqdm(range(len(gt_files) - 1), desc="Processing frames")

    with torch.no_grad():
        from torch.cuda.amp import autocast

        with autocast():
            torch.cuda.empty_cache()

            for i in progress_bar:
                # print(os.path.join(gt_folder, gt_files[i]), os.path.join(test_folder, test_files[i]))
                # exit()
                img1_gt, alpha1 = load_rgba_image(os.path.join(gt_folder, gt_files[i]), True)
                img2_gt, alpha2 = load_rgba_image(os.path.join(gt_folder, gt_files[i+1]), True)
                img1_test = load_rgba_image(os.path.join(test_folder, test_files[i]))
                img2_test = load_rgba_image(os.path.join(test_folder, test_files[i+1]))
                
                img1_test *= alpha1
                img2_test *= alpha2
                # import matplotlib.pyplot as plt

                # plt.subplot(1, 2, 1)
                # plt.imshow(img1_gt.permute(1,2,0).cpu())
                # plt.title(f"GT {gt_files[i]}")

                # plt.subplot(1, 2, 2)
                # plt.imshow(img1_test.permute(1,2,0).cpu())
                # plt.title(f"Test {test_files[i]}")
                # plt.show()
                # exit()
                flow_gt = compute_raft_flow(img1_gt, img2_gt, model, device)
                flow_test = compute_raft_flow(img1_test, img2_test, model, device)

                mse = mse_between_flows(flow_gt, flow_test)
                # if mse < 1.:
                mses.append(mse)

                # Show running average in progress bar
                avg_mse = np.mean(mses)
                progress_bar.set_postfix(avg_mse=f"{avg_mse:.8f}")

                # Free up memory
                del img1_gt, img2_gt, img1_test, img2_test, flow_gt, flow_test
                torch.cuda.empty_cache()

    return np.mean(mses)


# Paths
scenes=[ "Pony", "Piano"] #"Pony",
# scenes=["Pony"]

models = ["fg_loss", "unifiedH"]#, "unifieddyn4_nostaticdupe"] #, "bg_loss",
# models = [ "unifieddyn4_nostaticdupe"] #, "bg_loss",

final_res = {}
for scene in scenes:
    for mod in models:

        gt_folder = f"/home/barry/Desktop/PhD/SparseViewPaper/SuppMat_workingdir/Flow/GT_{scene}"
        test_folder = "output/Condense/"+scene+"/"+mod+"/masked/"

        op = optical_flow_mse(test_folder, gt_folder, model)
        final_res[f"{mod}-{scene}"] = float(op)
        print(f'{mod}-{scene}: {float(op):4f}')
        
import json 
with open("./optical_flow.json", "w") as fp:
    json.dump(final_res, fp, indent=4)
    