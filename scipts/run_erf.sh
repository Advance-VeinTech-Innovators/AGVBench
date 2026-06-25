#!/usr/bin/env bash
# Simple ERF Visualization Launcher
# Usage: ./run_erf.sh config.py checkpoint.pth [options]
set -x

CONFIG=$1
CHECKPOINT=$2
IMAGE_PATH=$3
NAME=$4

# Run ERF visualization
python tools/visualizations/vis_erf.py \
    $CONFIG \
    $CHECKPOINT \
    --image-path $IMAGE_PATH \
    --input-size 1024 \
    --save-path "./work_dirs/erf/${NAME}.png"