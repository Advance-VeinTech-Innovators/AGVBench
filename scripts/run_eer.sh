#!/usr/bin/env bash
# Draw ROC from fpr/tpr npy. bash run_eer.sh $WORK_DIRS $NAME $MODEL_NAMES
set -x

WORK_DIRS="work_dirs/_starlknet/vera220/baseline"
NAME="vera220_baseline"
MODEL_NAMES="StarLKNet-base StarLKNet-small"

# Draw ROC from fpr/tpr npy
python tools/visualizations/vis_eer.py \
    plot_curve \
    --work_dirs $WORK_DIRS \
    --name $NAME \
    --model_names $MODEL_NAMES \
    --smooth \
    --window_size 15