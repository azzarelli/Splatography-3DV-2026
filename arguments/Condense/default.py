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
     'resolution': [512, 150],
     'wavelevel':2
    },

    net_width = 128,
    
    plane_tv_weight = 0.0005,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    opacity_lambda = 0.,
    
    minview_weight=0.,
    tvtotal1_weight=0.0, #0001,
    spsmoothness_weight=0., #1,
    minmotion_weight=0.0, #0001,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=8000,
    coarse_iterations=3000,
    batch_size=2, # Was 4

    densify_from_iter=3000, #best at 3001
    densify_until_iter=10_000,
    densification_interval=100,
    densify_grad_threshold=0.001,
    opacity_reset_interval=3000,    

    pruning_interval=100,
    pruning_from_iter=3000,
    lambda_dssim = 0., #0.1,
    
    feature_lr=0.001
)