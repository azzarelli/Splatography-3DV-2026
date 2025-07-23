ModelHiddenParams = dict(
    scene_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [256, 50],
     'wavelevel':2
    },
    target_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [256, 150],
     'wavelevel':2
    },

    net_width = 128,
    
    plane_tv_weight = 0.0005,
    tv_background_weight = 0.0001,
    time_smoothness_weight = 0.0001,
    ts_background_weight   = 0.0001,
    l1_time_planes       = 0.0001,
    l1_col_planes_weight = 0.0001,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=8000,
    coarse_iterations=3000,
    batch_size=2, # Was 4
    opacity_reset_interval=3000,    
    feature_lr=0.005
)