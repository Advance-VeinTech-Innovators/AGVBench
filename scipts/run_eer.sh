#!/usr/bin/env bash
# Draw ROC from fpr/tpr npy. bash run_eer.sh $WORK_DIRS $NAME $MODEL_NAMES
set -x

WORK_DIRS="/yuchang/lsy_jx/AGVBench/work_dirs/_starlknet_eer_roc/tju600/starmixup"
NAME="tju600_starmixup"
MODEL_NAMES="StarLKNet-base StarLKNet-small"

# Draw ROC from fpr/tpr npy
python tools/visualizations/vis_eer.py \
    plot_curve \
    --work_dirs $WORK_DIRS \
    --name $NAME \
    --model_names $MODEL_NAMES \
    --smooth \
    --window_size 15