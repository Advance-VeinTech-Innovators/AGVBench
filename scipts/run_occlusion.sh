#!/usr/bin/env bash
# Run occlusion robustness experiments. bash run_occlusion.sh $CHECKPOINT $CONFIG $MAX_RATIO
set -x

# CONFIG=$1
# CHECKPOINT=$2
# MAX_RATIO=$3

CONFIG="configs/classification/_starlknet_ok/tju600/vgg16/vgg16_vanilla_sz224_bs32.py"
CHECKPOINT="work_dirs/classification/_starlknet_ok/tju600/vgg16/vgg16_vanilla_sz224_bs32/epoch_600.pth"
MAX_RATIO=1.0

python tools/analysis_tools/occlusion_robustness.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --max_ratio $MAX_RATIO \