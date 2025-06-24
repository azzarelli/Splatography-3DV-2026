#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=$2


if [ "$2" == "-1" ];then
  echo "Input 1 is the expname; Input 2 is [coffee, spinach, cut, flame, salmon, sear]"
  exit 1
fi

SAVEDIR=$1
# ARGS=tv001.py
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/TV001_$EXP_NAME" --configs arguments/Condense/$ARGS --test_iterations 1000
ARGS=tv005.py
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/TV005_$EXP_NAME" --configs arguments/Condense/$ARGS --test_iterations 1000
ARGS=tv0001.py
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/TV0001_$EXP_NAME" --configs arguments/Condense/$ARGS --test_iterations 1000
# ARGS=l1tp0001.py
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/0001_$EXP_NAME" --configs arguments/Condense/$ARGS --test_iterations 1000
# ARGS=l1tp00001.py
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/L100001_$EXP_NAME" --configs arguments/Condense/$ARGS --test_iterations 1000
