ModelHiddenParams = dict(
    scene_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [512, 50],
     'wavelevel':3
    },

    net_width = 128,
    
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0005,
    opacity_lambda = 0.,
    
)
OptimizationParams = dict(
    dataloader=True,
    iterations=10000,
    coarse_iterations=2000,
    batch_size=4, # Was 4

    densify_from_iter=3000, #best at 3001
    densify_until_iter=7_000,
    densification_interval=100,
    densify_grad_threshold=0.0001,
    opacity_reset_interval = 3000,    

    pruning_interval = 100,
    pruning_from_iter=3000,
    lambda_dssim = 0., #0.1,
    
    feature_lr=0.0025
)