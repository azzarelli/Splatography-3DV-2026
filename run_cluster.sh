#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=$2


if [ "$2" == "-1" ];then
  echo "Input 1 is the expname; Input 2 is [coffee, spinach, cut, flame, salmon, sear]"
  exit 1
fi

if [ "$1" == "spinach" ]; then
  echo "---- Cook Spinach ----"
  SAVEDIR="cook_spinach"
  ARGS=cook_spinach.py
  EVAL_LIST="0 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19"
  EVAL_LIST="0 2 3 4 5 6 7 8 11 12 13 14 15 16 17 18"
elif [ "$1" == "flame" ]; then
  echo "---- Flame Steak ----"
  SAVEDIR="flame_steak"
  ARGS=flame_steak.py
  EVAL_LIST="0 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19"
elif [ "$1" == "salmon" ]; then
  echo "---- Flame Salmon ----"
  SAVEDIR="flame_salmon"
  ARGS=flame_salmon_1.py
  EVAL_LIST="" # TODO - THIS hasnt been loaded on work pc
else
  echo "---- Unknown ----"
  exit 1
fi

echo "Training starting..."

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/default.py --test_iterations 1000
