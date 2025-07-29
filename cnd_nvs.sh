#!/bin/bash

# SAVEDIR=Bassist
# EXP_NAME=unifieddyn4_nostaticdupe
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test
# SAVEDIR=Piano
# EXP_NAME=unifieddyn4_nostaticdupe2
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test
# SAVEDIR=Pony
# EXP_NAME=unifieddyn4_nostaticdupe
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test
# SAVEDIR=Fruit
# EXP_NAME=unifieddyn4_nostaticdupe
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test
SAVEDIR=Piano
EXP_NAME=unifiedH
# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test

SAVEDIR=Pony
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test

SAVEDIR=Bassist
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test

SAVEDIR=Fruit
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test

SAVEDIR=Curling
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python generate_novel_views.py -s /media/barry/56EA40DEEA40BBCD/DATA/Condense/$SAVEDIR/ --expname "Condense/$SAVEDIR/$EXP_NAME" --configs arguments/Condense/default.py --start_checkpoint 11999 --view-test
