#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=unifieddyn4

SAVEDIR="flame_steak"
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/default.py
# SAVEDIR="flame_salmon"
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/salmon.py
# SAVEDIR="cook_spinach"
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/default.py

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/SC001$EXP_NAME" --configs arguments/dynerf/SC001.py --test_iterations 1000
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/SC0005$EXP_NAME" --configs arguments/dynerf/SC0005.py --test_iterations 1000
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/OL01$EXP_NAME" --configs arguments/dynerf/OL01.py --test_iterations 1000
