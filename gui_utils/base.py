import dearpygui.dearpygui as dpg
import numpy as np
import os
import copy
import psutil
import torch
from gaussian_renderer import render

class GUIBase:
    """This method servers to intialize the DPG visualization (keeping my code cleeeean!)
    
        Notes:
            none yet...
    """
    def __init__(self, gui, scene, gaussians, runname, view_test):
        
        self.gui = gui
        self.scene = scene
        self.gaussians = gaussians
        self.runname = runname
        self.view_test = view_test
        
        # Set the width and height of the expected image
        self.W, self.H = self.scene.getTestCameras()[0].image_width, self.scene.getTestCameras()[0].image_height
        self.fov = (self.scene.getTestCameras()[0].FoVy, self.scene.getTestCameras()[0].FoVx)

        if self.H > 800:
            self.W = self.W//2
            self.H = self.H //2
        # Initialize the image buffer
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        
        # Other important visualization parameters
        self.time = 0.
        self.show_radius = 30.
        self.vis_mode = 'render'
        self.show_dynamic = 0.
        self.w_thresh = 0.
        self.h_thresh = 0.
        
        self.set_w_flag = False
        self.w_val = 0.01
        # Set-up the camera for visualization
        self.show_scene_target = 0

        self.finecoarse_flag = True        
        self.switch_off_viewer = False
        self.switch_off_viewer_args = False
        self.full_opacity = False
        
        self.N_pseudo = 3 
        if view_test:
            if self.scene.dataset_type == 'dynerf':
                self.free_cams = [self.scene.getTestCameras()[0], self.scene.getTestCameras()[300]] + [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]
            else:
                self.free_cams = [self.scene.getTestCameras()[i*300] for i in range(4)] + [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]

        else:
            if self.scene.dataset_type == 'dynerf':
                self.free_cams = [self.scene.getTestCameras()[0]] + [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]
            else:
                self.free_cams = [self.scene.getTestCameras()[i*300] for i in range(4)] + [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]

            # try:
            #     self.free_cams = [self.scene.get_pseudo_view() for i in range(self.N_pseudo)]+ [self.scene.getTestCameras()[0]] + [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]
            # except:
            #     self.free_cams = [self.scene.getTestCameras()[0]] + [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]
        # self.free_cams = [self.scene.get_pseudo_view() for i in range(self.N_pseudo)] + [self.scene.getTrainCameras()[idx] for idx in self.scene.train_camera.zero_idxs]
        self.current_cam_index = 0
        self.original_cams = [copy.deepcopy(cam) for cam in self.free_cams]
        
        if self.gui:
            print('DPG loading ...')
            dpg.create_context()
            self.register_dpg()
    
    def __del__(self):
        dpg.destroy_context()

    
    def track_cpu_gpu_usage(self, time):
        # Print GPU and CPU memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 ** 2)  # Convert to MB

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        print(
            f'[{self.stage} {self.iteration}] Time: {time:.2f} | Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB | CPU Memory Usage: {memory_mb:.2f} MB')
    
    def render(self):
        if self.gui:
            tested = True
            while dpg.is_dearpygui_running():
                if self.iteration > self.final_iter and self.stage == 'coarse':
                    self.stage = 'fine'
                    self.init_taining()

                if self.view_test == False:
                    if self.iteration <= self.final_iter:
                        # Train the background seperately
                        if self.stage == 'coarse':
                            self.train_background_step()
                            self.train_foreground_step()
                        else:
                            # if self.iteration < (self.final_iter //2):
                            #     self.train_foreground_fine_step()
                            self.train_step()
                            self.train_depth_step()
                        self.iteration += 1


                    if (self.iteration % 4000) == 0:
                        if self.stage == 'fine':
                            self.test_step()

                    if self.iteration > self.final_iter and self.stage == 'fine':
                        self.stage = 'done'
                        dpg.stop_dearpygui() 
                        # exit()
                elif tested:
                    # self.test_step()
                    tested = False

                # self.render_video_step()
                # exit()
                
                
                # if self.view_test == True:
                #     self.train_depth()
                # with torch.no_grad():
                #     self.test_step()
                #     exit()
                with torch.no_grad():
                    self.viewer_step()
                    dpg.render_dearpygui_frame()
            dpg.destroy_context() 
        else:
            while self.stage != 'done':
                if self.iteration % 100 == 0:
                    print(f'[{self.stage}] {self.iteration}')
                if self.iteration > self.final_iter and self.stage == 'coarse':
                    self.stage = 'fine'
                    self.init_taining()

                if self.iteration <= self.final_iter:
                    self.train_step()
                    self.iteration += 1


                if (self.iteration % self.args.test_iterations) == 0 or (self.iteration == 1 and self.stage == 'fine' and self.opt.coarse_iterations > 50):
                    if self.stage == 'fine':
                        self.test_step()

                if self.iteration > self.final_iter and self.stage == 'fine':
                    self.stage = 'done'
                    exit()
                    
    @torch.no_grad()
    def viewer_step(self):
        
        if self.switch_off_viewer == False:
            # self.viewpoint_stack = self.scene.getTrainCameras()
            
            # self.scene.getTrainCameras().dataset.get_mask = True
            # dyn_mask =  self.get_target_mask()
            # self.scene.getTrainCameras().dataset.get_mask = False

            cam = self.free_cams[self.current_cam_index]
            cam.time = self.time
            buffer_image = render(
                    cam,
                    self.gaussians, 
                    self.pipe, 
                    self.background, 
                    stage=self.stage,
                    view_args={
                        'show_mask':self.show_scene_target,
                        'full_opac':self.full_opacity,
                        'w_thresh':self.w_thresh,
                        'dx_thresh': self.show_dynamic,
                        'h_thresh':self.h_thresh,
                        "set_w":self.w_val,
                        "set_w_flag":self.set_w_flag,
                        "viewer_status":self.switch_off_viewer_args,
                        "vis_mode":self.vis_mode,
                        "finecoarse_flag":self.finecoarse_flag,
                    }
            )
            
            try:
                buffer_image = buffer_image["render"]
            except:
                print(f'Mode "{self.vis_mode}" does not work')
                buffer_image = buffer_image['render']
                
            # if buffer_image.shape[0] == 1:
            #     buffer_image = (buffer_image - buffer_image.min())/(buffer_image.max() - buffer_image.min())
            #     buffer_image = buffer_image.repeat(3,1,1)
            buffer_image = torch.nn.functional.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H,self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

        buffer_image = self.buffer_image

        dpg.set_value(
            "_texture", buffer_image
        )  # buffer must be contiguous, else seg fault!
        
        # Add _log_view_camera
        if self.current_cam_index < self.N_pseudo:
            dpg.set_value("_log_view_camera", f"Random Novel Views")
        elif self.current_cam_index == self.N_pseudo:
            dpg.set_value("_log_view_camera", f"Test Views")
        else:
            dpg.set_value("_log_view_camera", f"Training Views")

    def save_scene(self):
        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
        self.scene.save(self.iteration, self.stage)
        print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
        torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + f"_{self.stage}_" + str(self.iteration) + ".pth")

    def register_dpg(self):
        
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=400,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            # ----------------
            #  Loss Functions
            # ----------------
            with dpg.group():
                if self.view_test is False:
                    dpg.add_text("Training info:")
                    dpg.add_text("no data", tag="_log_iter")
                    dpg.add_text("no data", tag="_log_loss")
                    dpg.add_text("no data", tag="_log_depth")
                    dpg.add_text("no data", tag="_log_opacs")
                    dpg.add_text("no data", tag="_log_dynscales")
                    dpg.add_text("no data", tag="_log_knn")

                    dpg.add_text("no data", tag="_log_points")
                else:
                    dpg.add_text("Training info: (Not training)")


            with dpg.collapsing_header(label="Testing info:", default_open=True):
                dpg.add_text("no data", tag="_log_psnr_test")
                dpg.add_text("no data", tag="_log_ssim")

            # ----------------
            #  Control Functions
            # ----------------
            with dpg.collapsing_header(label="Rendering", default_open=True):
                def callback_toggle_show_rgb(sender):
                    self.switch_off_viewer = ~self.switch_off_viewer
                def callback_toggle_use_controls(sender):
                    self.switch_off_viewer_args = ~self.switch_off_viewer_args
                def callback_toggle_finecoarse(sender):
                    self.finecoarse_flag = False if self.finecoarse_flag else True
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Render On/Off", callback=callback_toggle_show_rgb)
                    dpg.add_button(label="Ctrl On/Off", callback=callback_toggle_use_controls)
                    dpg.add_button(label="Fine/Coarse", callback=callback_toggle_finecoarse)

                     
                def callback_toggle_reset_cam(sender):
                    self.current_cam_index = 0
                    
                def callback_toggle_next_cam(sender):
                    self.current_cam_index = (self.current_cam_index + 1) % len(self.free_cams)
                    
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset cam", callback=callback_toggle_reset_cam)
                    dpg.add_text("no data", tag="_log_view_camera")
                    dpg.add_button(label="Next cam", callback=callback_toggle_next_cam)

                
                def callback_toggle_show_target(sender):
                    self.show_scene_target = 1
                def callback_toggle_show_scene(sender):
                    self.show_scene_target = -1 
                def callback_toggle_show_full(sender):
                    self.show_scene_target = 0 
                    
                def callback_toggle_reset_cam(sender):
                    for i in range(len(self.free_cams)):
                        self.free_cams[i] = copy.deepcopy(self.original_cams[i])
                    self.current_cam_index = 0
            
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Target", callback=callback_toggle_show_target)
                    dpg.add_button(label="Scene", callback=callback_toggle_show_scene)
                    dpg.add_button(label="Full", callback=callback_toggle_show_full)
                    dpg.add_button(label="Rst Fov", callback=callback_toggle_reset_cam)
                
                def callback_toggle_show_rgb(sender):
                    self.vis_mode = 'render'
                def callback_toggle_show_depth(sender):
                    self.vis_mode = 'D'
                def callback_toggle_show_edepth(sender):
                    self.vis_mode = 'ED'
                def callback_toggle_show_norms(sender):
                    self.vis_mode = 'norms'
                def callback_toggle_show_alpha(sender):
                    self.vis_mode = 'alpha'
                with dpg.group(horizontal=True):
                    dpg.add_button(label="RGB", callback=callback_toggle_show_rgb)
                    dpg.add_button(label="D", callback=callback_toggle_show_depth)
                    dpg.add_button(label="ED", callback=callback_toggle_show_edepth)
                    dpg.add_button(label="Norms", callback=callback_toggle_show_norms)
                    dpg.add_button(label="Alpha", callback=callback_toggle_show_alpha)
                
                def callback_toggle_show_xyz(sender):
                    self.vis_mode = 'xyz'
                def callback_toggle_show_dxyz1(sender):
                    self.vis_mode = 'dxyz_1'
                def callback_toggle_show_dxyz3(sender):
                    self.vis_mode = 'dxyz_3' 
                    
                with dpg.group(horizontal=True):
                    dpg.add_button(label="XYZ", callback=callback_toggle_show_xyz)
                    dpg.add_button(label="dX1", callback=callback_toggle_show_dxyz1)
                    dpg.add_button(label="dX3", callback=callback_toggle_show_dxyz3)
                
                def callback_toggle_show_extra(sender):
                    self.vis_mode = 'extra' 
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Extra", callback=callback_toggle_show_extra)
                
                def callback_toggle_show_fullopac(sender):
                    self.full_opacity = ~self.full_opacity
                def callback_toggle_show_wthesh(sender):
                    self.set_w_flag = ~self.set_w_flag
                    
                with dpg.group(horizontal=True):
                    dpg.add_button(label="h=1", callback=callback_toggle_show_fullopac)
                    dpg.add_button(label="Set/Unset w", callback=callback_toggle_show_wthesh)

                
                def callback_show_max_radius(sender):
                    self.show_radius = dpg.get_value(sender)
                    
                dpg.add_slider_float(
                    label="Show Radial Distance",
                    default_value=30.,
                    max_value=50.,
                    min_value=0.,
                    callback=callback_show_max_radius,
                )
                
                def callback_speed_control(sender):
                    self.time = dpg.get_value(sender)
                    
                dpg.add_slider_float(
                    label="Time",
                    default_value=0.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_speed_control,
                )
                
                def callback_toggle_view_dynamic(sender):
                    self.show_dynamic = dpg.get_value(sender)
                    
                dpg.add_slider_float(
                    label="dx > thresh",
                    default_value=1.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_toggle_view_dynamic,
                )
                def callback_toggle_h_thresh(sender):
                    self.h_thresh = dpg.get_value(sender)
                    
                dpg.add_slider_float(
                    label="h > thresh",
                    default_value=0.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_toggle_h_thresh,
                )
                
                def callback_toggle_w_thresh(sender):
                    self.w_thresh = dpg.get_value(sender)
                    
                dpg.add_slider_float(
                    label="w > thresh",
                    default_value=0.,
                    max_value=3.,
                    min_value=0.,
                    callback=callback_toggle_w_thresh,
                )
                
                def callback_toggle_w(sender):
                    self.w_val = dpg.get_value(sender)
                    
                dpg.add_slider_float(
                    label="Set globa w",
                    default_value=1.,
                    max_value=100.,
                    min_value=0.,
                    callback=callback_toggle_w,
                )
                
            def zoom_callback_fov(sender, app_data):
                delta = app_data  # scroll: +1 = up (zoom in), -1 = down (zoom out)
                cam = self.free_cams[self.current_cam_index]

                zoom_scale = 0.95  # Smaller = faster zoom

                # Scale FoV within limits
                cam.FoVy *= zoom_scale if delta > 0 else 1 / zoom_scale
                cam.FoVx *= zoom_scale if delta > 0 else 1 / zoom_scale

                # Optional clamp to prevent weird values
                cam.FoVy = np.clip(cam.FoVy, np.radians(10), np.radians(120))
                cam.FoVx = np.clip(cam.FoVx, np.radians(10), np.radians(120))
                cam.update_projections()
                
            with dpg.handler_registry():
                dpg.add_mouse_wheel_handler(callback=zoom_callback_fov)
            
        dpg.create_viewport(
            title=f"{self.runname}",
            width=self.W + 400,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        
        
            
        dpg.setup_dearpygui()

        dpg.show_viewport()
        

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

