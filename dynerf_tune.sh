#!/bin/bash

# Assign the first argument to a variable
EXP_NAME=$1

if [ "$1" == "-1" ];then
  echo "Input 1 is the expname; Input 2 is [coffee, spinach, cut, flame, salmon, sear]"
  exit 1
fi


# 4 Training Cameras at extremities
python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/bench" --configs arguments/dynerf/tuning/bench.py --test_iterations 1000

#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/binoc_$EXP_NAME" --configs arguments/dynerf/tuning/bench.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/baseline0.5_$EXP_NAME" --configs arguments/dynerf/tuning/baseline0.5.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/baseline0.05_$EXP_NAME" --configs arguments/dynerf/tuning/baseline0.05.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/baseline1.0_$EXP_NAME" --configs arguments/dynerf/tuning/baseline1.0.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/gtloss_$EXP_NAME" --configs arguments/dynerf/tuning/gt_loss.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/lrc_loss_$EXP_NAME" --configs arguments/dynerf/tuning/LRC_loss.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/rl_loss_$EXP_NAME" --configs arguments/dynerf/tuning/rl_loss.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/comp_loss_$EXP_NAME" --configs arguments/dynerf/tuning/comp_loss.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/comp_loss_late_$EXP_NAME" --configs arguments/dynerf/tuning/comp_loss_late.py --test_iterations 1000
#
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/18k_gtloss_$EXP_NAME" --configs arguments/dynerf/tuning/gt_loss.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/18k_lrc_loss_$EXP_NAME" --configs arguments/dynerf/tuning/LRC_loss.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/18k_rl_loss_$EXP_NAME" --configs arguments/dynerf/tuning/rl_loss.py --test_iterations 1000
#python gui.py -s /data/dynerf/sear_steak/ --expname "dynerf/sear_steak_rect/18k_comp_loss_late_$EXP_NAME" --configs arguments/dynerf/tuning/comp_loss_late.py --test_iterations 1000