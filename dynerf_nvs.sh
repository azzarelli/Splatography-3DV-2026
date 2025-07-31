#!/bin/bash

EXP_NAME=unifieddyn4

# SAVEDIR="cook_spinach"
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/spinach.py --start_checkpoint 13999 --view-test


SAVEDIR="flame_salmon"
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/salmon.py --start_checkpoint 13999 --view-test


SAVEDIR="flame_steak"
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/steak.py --start_checkpoint 8000 --view-test
