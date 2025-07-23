#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=$2
DATA=$3


if [ "$2" == "-1" ];then
  echo "Input 1 is the expname; Input 2 is [coffee, spinach, cut, flame, salmon, sear]"
  exit 1
fi

if [ "$1" == "spinach" ]; then
  echo "---- Cook Spinach ----"
  SAVEDIR="cook_spinach"
  ARGS=cook_spinach.py
elif [ "$1" == "flame" ]; then
  echo "---- Flame Steak ----"
  SAVEDIR="flame_steak"
  ARGS=flame_steak.py
elif [ "$1" == "salmon" ]; then
  echo "---- Flame Salmon ----"
  SAVEDIR="flame_salmon"
  ARGS=flame_salmon_1.py
else
  echo "---- Unknown ----"
  exit 1
fi

echo "Training starting..."

python train.py -s $DATA/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/default.py
