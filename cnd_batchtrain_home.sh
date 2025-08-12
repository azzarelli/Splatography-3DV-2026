#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=4dgs_densification
ARGS=default.py

# SAVEDIR=Piano
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py
SAVEDIR=Fruit
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py
SAVEDIR=Curling
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py
SAVEDIR=Pony
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py
SAVEDIR=Bassist
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py 
