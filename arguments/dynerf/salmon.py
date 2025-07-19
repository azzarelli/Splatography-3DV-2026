ModelHiddenParams = dict(
    scene_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [128, 80],
     'wavelevel':2
    },
    target_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [128, 150],
     'wavelevel':2
    },

    net_width = 128,
    
    plane_tv_weight = 0.001,
    time_smoothness_weight = 0.0001,
    l1_time_planes =  0.001,
    minview_weight=0.001,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=14000,
    coarse_iterations=3000,
    batch_size=4, # Was 4
    
    opacity_reset_interval = 4000,    

    pruning_interval = 100,
    pruning_from_iter=3000,
    lambda_dssim = 0., #0.1,
    
    # feature_lr=0.0002 #0.0005 #0.0002
)

ModelHiddenParams = dict(
    scene_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [128, 25],
     'wavelevel':2
    },
    target_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 25],
     'wavelevel':2
    },

    net_width = 128,
    
    plane_tv_weight = 0.001,
    time_smoothness_weight = 0.0001,
    l1_time_planes =  0.001,
    minview_weight=0.001,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=14000,
    coarse_iterations=3000,
    batch_size=2, # Was 4
    
    opacity_reset_interval = 4000,    

    pruning_interval = 100,
    pruning_from_iter=3000,
    lambda_dssim = 0., #0.1,
    
    # feature_lr=0.0002 #0.0005 #0.0002
)