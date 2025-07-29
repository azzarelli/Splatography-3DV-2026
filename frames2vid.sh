#!/bin/bash

FOLDER=masked
SAVEDIR=Piano
OUTPUT=test
DATSET=Condense
EXP_NAME=unifiedH
MEDIA="output/$DATSET/$SAVEDIR/$EXP_NAME/$FOLDER"
ffmpeg -framerate 30 -i output/$DATSET/$SAVEDIR/$EXP_NAME/$FOLDER/%d.png \
-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
-c:v libx264 -pix_fmt yuv420p \
output/$DATSET/$SAVEDIR/$EXP_NAME/$OUTPUT.mp4