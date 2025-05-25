#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=$2


if [ "$2" == "-1" ];then
  echo "Input 1 is the expname; Input 2 is [coffee, spinach, cut, flame, salmon, sear]"
  exit 1
fi

SAVEDIR=$1
ARGS=default.py

if [ "$3" == "render" ]; then
  echo "Rendering process starting..."
  python render.py --model_path "output/Condense/$SAVEDIR/$EXP_NAME" --skip_train --configs arguments/Condense/default.py
elif [ "$3" == "view" ]; then
  echo "Viewing..."

  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000 --start_checkpoint $4 --view-test
elif [ "$3" == "ext" ]; then
  echo "Extending..."

  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000 --start_checkpoint 7000 
elif [ "$3" == "skip-coarse" ]; then
  echo "Skip Coarse..."

  CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000 --skip-coarse $4
else
  echo "Training starting..."

  TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python gui.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --test_iterations 1000

fi
