import psutil
import torch
from tqdm import tqdm

class GUIBase:
    def __init__(self, gui, scene, gaussians, runname, view_test):
        
        self.gui = gui
        self.scene = scene
        self.gaussians = gaussians
        self.runname = runname
        self.view_test = view_test
        
        # Set the width and height of the expected image
        self.W, self.H = self.scene.getTestCameras()[0].image_width, self.scene.getTestCameras()[0].image_height
        self.fov = (self.scene.getTestCameras()[0].FoVy, self.scene.getTestCameras()[0].FoVx)
        
    def track_cpu_gpu_usage(self, time):
        # Print GPU and CPU memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 ** 2)  # Convert to MB

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        print(
            f'[{self.stage} {self.iteration}] Time: {time:.2f} | Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB | CPU Memory Usage: {memory_mb:.2f} MB')

    def train(self):
        """Train without gui"""
        pbar = tqdm(initial=0, total=self.final_iter, desc=f"[{self.stage}]")

        while self.stage != 'done':
            if self.iteration > self.final_iter and self.stage == 'coarse':
                self.stage = 'fine'
                self.init_taining()
                pbar = tqdm(initial=0, total=self.final_iter, desc=f"[{self.stage}]")

            if self.iteration <= self.final_iter:
                # Train background and/or foreground depending on stage
                if self.stage == 'coarse':
                    self.train_background_step()
                    self.train_foreground_step()
                else:
                    self.train_step()

                self.iteration += 1
                pbar.update(1)

            if self.iteration % 1000 == 0 and self.stage == 'fine':
                self.test_step()

            if self.iteration > self.final_iter and self.stage == 'fine':
                self.stage = 'done'
                break  # Exit the loop instead of calling exit()

        pbar.close()
        self.full_evaluation()

    def save_scene(self):
        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
        self.scene.save(self.iteration, self.stage)
        
        self.gaussians.save_checkpoint(self.iteration, self.stage, self.scene.model_path)
  
def get_in_view_dyn_mask(camera, xyz: torch.Tensor) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # Apply full projection (world → clip space)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    # Homogeneous divide to get NDC coordinates
    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    # Visibility check
    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2].abs() <= 1)
    visible_mask = in_ndc_bounds & in_front
    
    # Compute pixel coordinates
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long()
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long()    # Init mask values
    mask_values = torch.zeros(N, dtype=torch.bool, device=device)

    # Only sample pixels for visible points
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]

    if valid_idx.numel() > 0:
        px_valid = px[valid_idx].clamp(0, camera.image_width - 1)
        py_valid = py[valid_idx].clamp(0, camera.image_height - 1)
        mask = camera.mask.to(device)
        sampled_mask = mask[py_valid, px_valid]  # shape: [#valid]
        mask_values[valid_idx] = sampled_mask.bool()
    # import matplotlib.pyplot as plt

    # # Assuming tensor is named `tensor_wh` with shape [W, H]
    # # Convert to [H, W] for display (matplotlib expects H first)
    # mask[py_valid, px_valid] = 0.5
    # print(py_valid.shape)

    # tensor_hw = mask.cpu()  # If it's on GPU
    # plt.imshow(tensor_hw, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # exit()
    return mask_values.long()

