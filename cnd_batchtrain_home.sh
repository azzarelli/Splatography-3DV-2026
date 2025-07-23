#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=ffff
ARGS=default.py

# SAVEDIR=Pony
# python render.py --model_path "output/Condense/$SAVEDIR/$EXP_NAME" --skip_train --configs arguments/Condense/default.py
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000
# CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/  --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test
# 
# SAVEDIR=Pony
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

# SAVEDIR=Bassist
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

SAVEDIR=Piano
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/FLR01$EXP_NAME" --configs arguments/Condense/FLR01.py --test_iterations 1000
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/FLR005$EXP_NAME" --configs arguments/Condense/FLR005.py --test_iterations 1000
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/FLR0025$EXP_NAME" --configs arguments/Condense/FLR0025.py --test_iterations 1000
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/FLR001$EXP_NAME" --configs arguments/Condense/FLR001.py --test_iterations 1000
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/FLR0005$EXP_NAME" --configs arguments/Condense/FLR0005.py --test_iterations 1000

# SAVEDIR=Curling
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

# SAVEDIR=Fruit
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000


