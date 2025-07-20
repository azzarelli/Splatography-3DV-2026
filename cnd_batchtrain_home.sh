#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=final1
ARGS=default.py

SAVEDIR=Fruit
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

SAVEDIR=Pony
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

# SAVEDIR=Bassist
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

# SAVEDIR=Curling
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

# SAVEDIR=Pianist
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

