#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=$1

if [ "$1" == "-1" ];then
  echo "Input 1 is the expname; Input 2 is [coffee, spinach, cut, flame, salmon, sear]"
  exit 1
fi

if [ "$2" == "coffee" ]; then
  echo "---- Coffee Martini ----"
  SAVEDIR="coffee_martini"
  ARGS=coffee_martini.py
  EVAL_LIST="0 2 3 4 5 6 7 8 11 12 13 14 15 16"
elif [ "$2" == "spinach" ]; then
  echo "---- Cook Spinach ----"
  SAVEDIR="cook_spinach"
  ARGS=cook_spinach.py
  EVAL_LIST="0 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19"
elif [ "$2" == "cut" ]; then
  echo "---- Cut Roasted Beef ----"
  SAVEDIR="cut_roasted_beef"
  ARGS=cut_roasted_beef.py
  EVAL_LIST="0 2 3 4 5 6 7 8 11 12 13 14 15 16 17 18"
elif [ "$2" == "flame" ]; then
  echo "---- Flame Steak ----"
  SAVEDIR="flame_steak"
  ARGS=flame_steak.py
  EVAL_LIST="0 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19"
elif [ "$2" == "salmon" ]; then
  echo "---- Flame Salmon ----"
  SAVEDIR="flame_salmon"
  ARGS=flame_salmon_1.py
  EVAL_LIST="" # TODO - THIS hasnt been loaded on work pc
elif [ "$2" == "sear" ]; then
  echo "---- Sear Steak ----"
  SAVEDIR="sear_steak"
  ARGS=sear_steak.py
  EVAL_LIST="0 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19"

else
  echo "---- Unknown ----"
  exit 1
fi

SAVEDIR="flame_steak"
ARGS=flame_steak.py
python gui.py -s /home/barry/data/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/default.py --test_iterations 1000


# python gui.py -s /home/barry/data/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/default.py --test_iterations 1000
# python gui.py -s /home/barry/data/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/dssim2.py --test_iterations 1000

# SAVEDIR="coffee_martini"
# ARGS=coffee_martini.py
# 4 Training Cameras at extremities
# python gui.py -s /home/barry/data/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/l1005$EXP_NAME" --configs arguments/dynerf/l1005.py --test_iterations 1000

# SAVEDIR="cut_roasted_beef"
# ARGS=cut_roasted_beef.py
# python gui.py -s /home/barry/data/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/l1005$EXP_NAME" --configs arguments/dynerf/l1005.py --test_iterations 1000

# SAVEDIR="sear_steak"
#   ARGS=sear_steak.py
# python gui.py -s /home/barry/data/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/l1005$EXP_NAME" --configs arguments/dynerf/l1005.py --test_iterations 1000

# SAVEDIR="sear_steak"
# ARGS=sear_steak.py
# EVAL_LIST="0 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19"
# python gui.py -s /home/barry/data/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/$EXP_NAME" --configs arguments/dynerf/default.py --test_iterations 1000


# python gui.py -s /home/barry/data/dynerf/$SAVEDIR/ --expname "dynerf/$SAVEDIR/halfref_$EXP_NAME" --configs arguments/dynerf/default2.py --test_iterations 1000
# python render.py --model_path "output/dynerf/$SAVEDIR/$EXP_NAME" --skip_train --configs arguments/dynerf/default.py
# Metrics
#python render.py --model_path "output/dynerf/$SAVEDIR/$EXP_NAME" --skip_train --configs arguments/dynerf/$ARGS