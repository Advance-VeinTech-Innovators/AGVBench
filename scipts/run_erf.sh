#!/usr/bin/env bash

# Simple ERF Visualization Launcher
# Usage: ./run_erf.sh config.py checkpoint.pth [options]

set -x

# CONFIG=$1
# CHECKPOINT=$2

CONFIG="configs/classification/_starlknet_ok/tju600/r18/r18_saliencymix_sz224_bs32.py"
CHECKPOINT="work_dirs/classification/_starlknet_ok/tju600/r18/r18_saliencymix_sz224_bs32/epoch_600.pth"

IMAGE_PATH="demo/tju600_24.bmp"

# Run ERF visualization
python tools/visualizations/vis_erf.py \
    $CONFIG \
    $CHECKPOINT \
    --image-path $IMAGE_PATH \
    --input-size 1024 \
    --save-path "./work_dirs/erf/erf_heatmap_wtxgrn_b.png"