#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=$2


if [ "$2" == "-1" ];then
  echo "Input 1 is the expname; Input 2 is [coffee, spinach, cut, flame, salmon, sear]"
  exit 1
fi

if [ "$1" == "coffee" ]; then
  echo "---- Coffee Martini ----"
  SAVEDIR="coffee_martini"
  ARGS=coffee_martini.py
  EVAL_LIST="0 2 3 4 5 6 7 8 11 12 13 14 15 16"
elif [ "$1" == "spinach" ]; then
  echo "---- Cook Spinach ----"
  SAVEDIR="cook_spinach"
  ARGS=cook_spinach.py
  EVAL_LIST="0 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19"
elif [ "$1" == "cut" ]; then
  echo "---- Cut Roasted Beef ----"
  SAVEDIR="cut_roasted_beef"
  ARGS=cut_roasted_beef.py
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
elif [ "$1" == "sear" ]; then
  echo "---- Sear Steak ----"
  SAVEDIR="sear_steak"
  ARGS=sear_steak.py
  EVAL_LIST="0 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19"

else
  echo "---- Unknown ----"
  exit 1
fi

echo "Training starting..."

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/SC01$EXP_NAME" --configs arguments/dynerf/SC01.py --test_iterations 1000
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/SC001$EXP_NAME" --configs arguments/dynerf/SC001.py --test_iterations 1000
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/SC0005$EXP_NAME" --configs arguments/dynerf/SC0005.py --test_iterations 1000
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py -s /media/barry/56EA40DEEA40BBCD/DATA/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/OL01$EXP_NAME" --configs arguments/dynerf/OL01.py --test_iterations 1000
