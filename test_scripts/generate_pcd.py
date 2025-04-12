


import numpy as np
import sys
import torch
import sys
import random
from utils.general_utils import safe_state

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

import matplotlib.pyplot as plt
import open3d as o3d

from DAV2.depth_anything_v2.dpt import DepthAnythingV2


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True




class PointCloudGenerator():
    
    def init_depth_anything_v2(self):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to(DEVICE).eval()
        return model
    
    def __init__(self,
        args, 
        hyperparams, 
        dataset, 
        opt, 
        pipe,
        expname
    ):        
        self.args = args
        self.dataset = dataset
        self.hyperparams = hyperparams
        self.opt = opt
        self.pipe = pipe
        self.expname = expname
        
        from scene import Scene, GaussianModel
        self.gaussians = GaussianModel(dataset.sh_degree, hyperparams)
        self.scene = Scene(dataset, self.gaussians)

        self.depth_model = self.init_depth_anything_v2()
    
        
    def run(self):
        """Extracts point clouds and launches interactive Dash viewer for live depth scaling."""
        self.viewpoint_stack = self.scene.getTrainCameras()
        first_frame = [self.viewpoint_stack[i * 300] for i in range(4)]

        import plotly.graph_objects as go
        import dash
        from dash import dcc, html, Input, Output
        import numpy as np

        camera_data = []
        for idx, view in enumerate(first_frame):
            # Get raw depth
            depth = self.depth_model.infer_image(view.original_image.cuda())
            # Invert if necessary
            depth = (depth - depth.min()) * -1 + depth.max()

            # Get camera-space points and RGB
            xyz_cam, rgb = view.project_depth_to_point_cloud(depth, downsample=8)
            cam_to_world = view.world_view_transform.inverse()
            camera_data.append({
                "xyz_cam": xyz_cam.cpu(),
                "colors": rgb.cpu(),
                "cam_to_world": cam_to_world.cpu(),
                "view":view
            })

        # ------------- DASH APP ------------- #
        app = dash.Dash(__name__)
        server = app.server

        app.layout = html.Div([
            html.H2("Live Depth Scaling Point Cloud Viewer"),

            html.Div([
                html.Div([
                    html.Label(f"Camera {i} Scale"),
                    dcc.Slider(id=f'scale-{i}', min=0.1, max=3.0, step=0.05, value=1.0,
                            tooltip={"placement": "bottom", "always_visible": True})
                ], style={'padding': '10px', 'width': '20%'}) for i in range(len(camera_data))
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

            dcc.Graph(id='pointcloud-graph', style={'height': '90vh'}),
        ])

        @app.callback(
            Output('pointcloud-graph', 'figure'),
            [Input(f'scale-{i}', 'value') for i in range(len(camera_data))]
        )
        def update_graph(*scales):
            traces = []
            for i, (scale, cam) in enumerate(zip(scales, camera_data)):
                xyz_cam = cam["xyz_cam"] * scale 
                N = xyz_cam.shape[0]
                xyz_h = torch.cat([xyz_cam, torch.ones(N, 1)], dim=1).T
                view = cam['view']
                xyz_world = (view.full_proj_transform.T @ xyz_h).T[:, :3]
                # xyz_world[:, 1:] *= -1 
                pts = xyz_world.numpy()
                colors = cam["colors"].numpy()

                traces.append(go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1.5,
                        color=['rgb({},{},{})'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors],
                        opacity=0.8
                    ),
                    name=f"Cam {i}"
                ))

                # Add camera position marker
                cam_origin = cam["cam_to_world"] @ torch.tensor([0, 0, 0, 1.0])
                traces.append(go.Scatter3d(
                    x=[cam_origin[0].item()],
                    y=[cam_origin[1].item()],
                    z=[cam_origin[2].item()],
                    mode='markers+text',
                    marker=dict(size=6, color='red', symbol='x'),
                    text=[f"Cam {i}"],
                    name=f"Camera {i} Pos"
                ))

            fig = go.Figure(data=traces)
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                showlegend=False
            )
            return fig

        app.run_server(debug=False, host=self.args.ip)
def display_image_and_depth_pair(tensor_rgb, tensor_gray):
    # Normalize grayscale tensor to [0, 1]
    tensor_gray= tensor_gray.cpu()
    tensor_gray_norm = (tensor_gray - tensor_gray.min()) / (tensor_gray.max() - tensor_gray.min())

    # Prepare tensors for display
    image_rgb = tensor_rgb.permute(1, 2, 0).numpy()
    image_gray = tensor_gray_norm.numpy()

    # Display side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(image_rgb)
    axs[0].set_title('RGB Tensor')
    axs[0].axis('off')

    axs[1].imshow(image_gray, cmap='gray')
    axs[1].set_title('Grayscale Tensor')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def display_torch_image(image):
    """Displays the torch tensor for us
    
        Args:
            image: torch.Tensor, shaped 3, H, W
        
    """
    image = image.permute(1, 2, 0).cpu().numpy()

    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    print("Generating point cloud for: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    gen = PointCloudGenerator(
        args=args, 
        hyperparams=hp.extract(args), 
        dataset=lp.extract(args), 
        opt=op.extract(args), 
        pipe=pp.extract(args),
        expname=args.expname
    )

    gen.run()