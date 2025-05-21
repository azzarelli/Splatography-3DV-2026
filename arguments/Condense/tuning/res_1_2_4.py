ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1, 2, 4],
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
    iterations = 7000,
    batch_size=1, # Was 4
    coarse_iterations = 0,

    densify_from_iter = 500,
    densify_until_iter = 10_000,
    densification_interval = 100,

    pruning_from_iter = 500,
    pruning_interval=100,

    opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.01,
    opacity_threshold_fine_init = 0.01,
    opacity_threshold_fine_after = 0.01,

    pcd_thresh = 0.01,
    pcd_k = 4,
)