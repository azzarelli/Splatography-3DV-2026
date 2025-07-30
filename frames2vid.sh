#!/bin/bash

FOLDER=masked
SAVEDIR=Piano
OUTPUT=test
DATSET=Condense
EXP_NAME=fg_loss

INPUT_DIR="output/$DATSET/$SAVEDIR/$EXP_NAME/$FOLDER"
WORK_DIR="output/$DATSET/$SAVEDIR/$EXP_NAME/jpg_frames"
OUTPUT_VIDEO="output/$DATSET/$SAVEDIR/$EXP_NAME/$OUTPUT.mp4"


# mkdir -p "$WORK_DIR"

# echo "Flattening PNGs to JPGs with black background..."

# shopt -s nullglob
# for img in "$INPUT_DIR"/*.png; do
#     fname=$(basename "$img" .png)

#     ffmpeg -y -i "$img" \
#     -f lavfi -i "color=black" \
#     -filter_complex "[1:v][0:v]scale2ref[bg][fg]; [bg][fg]overlay=format=auto" \
#     -frames:v 1 "$WORK_DIR/$fname.jpg"
# done

echo "Creating video from JPGs..."

ffmpeg -r 30 -i $WORK_DIR/%d.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $OUTPUT_VIDEO

echo "✅ Done! Video saved to: $OUTPUT_VIDEO"
