ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2],
    defor_depth = 0,
    net_width = 128,

    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,

    no_do=False,
    no_dshs=False,
    no_ds=False,

    render_process=False,
    static_mlp=False,

    bounds=2.5,

)
OptimizationParams = dict(
    dataloader=True,
    iterations = 4000,
    batch_size=4, # Was 4
    coarse_iterations = 0,

    densify_from_iter = 10,
    densification_interval = 40,
    densify_until_iter=2_000,

    pruning_from_iter = 100,
    pruning_interval=40,

    opacity_reset_interval = 1000,
    opacity_threshold_coarse = 0.01,
    opacity_threshold_fine_init = 0.01,
    opacity_threshold_fine_after = 0.01,

    pcd_thresh = 0.001,
    pcd_k = 4,

    lambda_dssim=0.1,

)