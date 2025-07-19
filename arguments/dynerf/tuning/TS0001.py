_base_ = './default.py'
ModelHiddenParams = dict(
    plane_tv_weight = 0.0001,
    time_smoothness_weight = 0.0001,
    l1_time_planes =  0.001,
    minview_weight=0.001,
)