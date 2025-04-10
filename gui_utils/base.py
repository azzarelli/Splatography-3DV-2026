import dearpygui.dearpygui as dpg
import numpy as np
import os
import copy
import psutil
import torch

class GUIBase:
    """This method servers to intialize the DPG visualization (keeping my code cleeeean!)
    
        Notes:
            none yet...
    """
    def __init__(self, gui, scene, gaussians):
        
        self.gui = gui
        self.scene = scene
        self.gaussians = gaussians
        
        # Set the width and height of the expected image
        self.W, self.H = self.scene.getTestCameras()[0].image_width, self.scene.getTestCameras()[0].image_height
        self.fov = (self.scene.getTestCameras()[0].FoVy, self.scene.getTestCameras()[0].FoVx)

        # Initialize the image buffer
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        
        # Other important visualization parameters
        self.time = 0.
        self.show_radius = 30.
        self.vis_mode = 'render'
        self.show_dynamic = 0.
        self.show_opacity = 10.
        
        # Set-up the camera for visualization

        self.cam =copy.deepcopy(self.scene.getTestCameras()[0])
        
        if self.gui:
            print('DPG loading ...')
            dpg.create_context()
            self.register_dpg()
    
    def __del__(self):
        dpg.destroy_context()

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

            with dpg.group():
                if self.checkpoint is None:
                    dpg.add_text("Training info:")
                    dpg.add_text("no data", tag="_log_iter")
                    dpg.add_text("no data", tag="_log_loss")
                    dpg.add_text("no data", tag="_log_depth")
                    dpg.add_text("no data", tag="_log_opacs")
                    dpg.add_text("no data", tag="_log_points")
                else:
                    dpg.add_text("Training info: (Not training)")


            with dpg.collapsing_header(label="Testing info:", default_open=True):
                dpg.add_text("no data", tag="_log_psnr_test")
                dpg.add_text("no data", tag="_log_ssim")


            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                
                def callback_toggle_show_rgb(sender):
                    self.vis_mode = 'render'
                def callback_toggle_show_depth(sender):
                    self.vis_mode = 'depth'
                def callback_toggle_show_alpha(sender):
                    self.vis_mode = 'alpha'
                def callback_toggle_show_norms(sender):
                    self.vis_mode = 'norms'
                    
                with dpg.group(horizontal=True):
                    dpg.add_button(label="RGB", callback=callback_toggle_show_rgb)
                    dpg.add_button(label="Depth", callback=callback_toggle_show_depth)
                    dpg.add_button(label="Alpha", callback=callback_toggle_show_alpha)
                    dpg.add_button(label="Normls", callback=callback_toggle_show_norms)
                    
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
                    label="Dynamic View Thresh",
                    default_value=1.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_toggle_view_dynamic,
                )
                def callback_toggle_view_opacity(sender):
                    self.show_opacity = dpg.get_value(sender)
                    
                dpg.add_slider_float(
                    label="Dynamic Opacity",
                    default_value=1.,
                    max_value=10.,
                    min_value=0.,
                    callback=callback_toggle_view_opacity,
                )

        dpg.create_viewport(
            title="WavePlanes",
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
            while dpg.is_dearpygui_running():
                if self.iteration > self.final_iter and self.stage == 'coarse':
                    self.stage = 'fine'
                    self.init_taining()

                if self.checkpoint is None:
                    if self.iteration <= self.final_iter:
                        self.train_step()
                        self.iteration += 1


                    if (self.iteration % self.args.test_iterations) == 0 or (self.iteration == 1 and self.stage == 'fine' and self.opt.coarse_iterations > 50):
                        if self.stage == 'fine':
                            self.test_step()

                    if self.iteration > self.final_iter and self.stage == 'fine':
                        self.stage = 'done'
                        exit()

                with torch.no_grad():
                    self.viewer_step()
                    dpg.render_dearpygui_frame()
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