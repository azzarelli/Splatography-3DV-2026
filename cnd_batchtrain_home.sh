#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=unifieddyn
ARGS=default.py

# SAVEDIR=Pony
# python render.py --model_path "output/Condense/$SAVEDIR/$EXP_NAME" --skip_train --configs arguments/Condense/default.py
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000
# CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/  --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test
# 
# SAVEDIR=Pony
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

SAVEDIR=Bassist
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

SAVEDIR=Pianist
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

SAVEDIR=Fruit
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000


# SAVEDIR=Curling
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

# SAVEDIR=Pianist
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

