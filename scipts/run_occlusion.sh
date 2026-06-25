#!/usr/bin/env bash
# Run occlusion robustness experiments. bash run_occlusion.sh $CHECKPOINT $CONFIG $MAX_RATIO
set -x

CONFIG=$1
CHECKPOINT=$2
MAX_RATIO=$3

python tools/analysis_tools/occlusion_robustness.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --max_ratio $MAX_RATIO \