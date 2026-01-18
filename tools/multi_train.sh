#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

BASE_DIR="/lisiyuan/jx/AGVBench/configs/classification/_starlknet_ok/vera220"

CFG_LIST=(
    # ResNet18
    # ${BASE_DIR}/r18/r18_vanilla_bs32_sz224.py
    # ${BASE_DIR}/r18/r18_mixup_bs32_sz224.py
    # ${BASE_DIR}/r18/r18_cutmix_bs32_sz224.py
    # ${BASE_DIR}/r18/r18_saliencymix_bs32_sz224.py
    # ${BASE_DIR}/r18/r18_starmix_bs32_sz224.py
    # ${BASE_DIR}/r18/r18_starmixplus_bs32_sz224.py
    # ResNet50
    # ${BASE_DIR}/r50/r50_vanilla_bs32_sz224.py
    # ${BASE_DIR}/r50/r50_mixup_bs32_sz224.py
    # ${BASE_DIR}/r50/r50_cutmix_bs32_sz224.py
    # ${BASE_DIR}/r50/r50_saliencymix_bs32_sz224.py
    # ${BASE_DIR}/r50/r50_starmix_bs32_sz224.py
    # ${BASE_DIR}/r50/r50_starmixplus_bs32_sz224.py
    # AMPVNet
    # ${BASE_DIR}/ampvnet/ampvnet_vanilla_sz224_bs32.py
    # ${BASE_DIR}/ampvnet/ampvnet_mixup_sz224_bs32.py
    # ${BASE_DIR}/ampvnet/ampvnet_cutmix_sz224_bs32.py
    # ${BASE_DIR}/ampvnet/ampvnet_saliencymix_sz224_bs32.py
    # ${BASE_DIR}/ampvnet/ampvnet_starmix_sz224_bs32.py
    # ${BASE_DIR}/ampvnet/ampvnet_starmixplus_sz224_bs32.py
    # FVRASNet
    ${BASE_DIR}/fvrasnet/fvrasnet_vanilla_sz224_bs32.py
    ${BASE_DIR}/fvrasnet/fvrasnet_mixup_sz224_bs32.py
    ${BASE_DIR}/fvrasnet/fvrasnet_cutmix_sz224_bs32.py
    ${BASE_DIR}/fvrasnet/fvrasnet_saliencymix_sz224_bs32.py
    ${BASE_DIR}/fvrasnet/fvrasnet_starmix_sz224_bs32.py
    ${BASE_DIR}/fvrasnet/fvrasnet_starmixplus_sz224_bs32.py
    # VGG16
    # ${BASE_DIR}/vgg16/vgg16_vanilla_sz224_bs32.py
    # ${BASE_DIR}/vgg16/vgg16_mixup_sz224_bs32.py
    # ${BASE_DIR}/vgg16/vgg16_cutmix_sz224_bs32.py
    # ${BASE_DIR}/vgg16/vgg16_saliencymix_sz224_bs32.py
    # ${BASE_DIR}/vgg16/vgg16_starmix_sz224_bs32.py
    # ${BASE_DIR}/vgg16/vgg16_starmixplus_sz224_bs32.py
    # FVCNN
    ${BASE_DIR}/fvcnn/fvcnn_vanilla_sz224_bs32.py
    ${BASE_DIR}/fvcnn/fvcnn_mixup_sz224_bs32.py
    ${BASE_DIR}/fvcnn/fvcnn_cutmix_sz224_bs32.py
    ${BASE_DIR}/fvcnn/fvcnn_saliencymix_sz224_bs32.py
    ${BASE_DIR}/fvcnn/fvcnn_starmix_sz224_bs32.py
    ${BASE_DIR}/fvcnn/fvcnn_starmixplus_sz224_bs32.py
    # PVCNN
    ${BASE_DIR}/pvcnn/pvcnn_vanilla_sz224_bs32.py
    ${BASE_DIR}/pvcnn/pvcnn_mixup_sz224_bs32.py
    ${BASE_DIR}/pvcnn/pvcnn_cutmix_sz224_bs32.py
    ${BASE_DIR}/pvcnn/pvcnn_saliencymix_sz224_bs32.py
    ${BASE_DIR}/pvcnn/pvcnn_starmix_sz224_bs32.py
    ${BASE_DIR}/pvcnn/pvcnn_starmixplus_sz224_bs32.py
    # WTxGRN Base
    ${BASE_DIR}/wtxgrn/wtxgrn_b_vanilla_sz224_bs32.py
    ${BASE_DIR}/wtxgrn/wtxgrn_b_mixup_sz224_bs32.py
    ${BASE_DIR}/wtxgrn/wtxgrn_b_cutmix_sz224_bs32.py
    ${BASE_DIR}/wtxgrn/wtxgrn_b_saliencymix_sz224_bs32.py
    ${BASE_DIR}/wtxgrn/wtxgrn_b_starmix_sz224_bs32.py
    ${BASE_DIR}/wtxgrn/wtxgrn_b_starmixplus_sz224_bs32.py
    # RSNet Local
    ${BASE_DIR}/rsnet/rsnet_local_vanilla_sz224_bs32.py
    ${BASE_DIR}/rsnet/rsnet_local_mixup_sz224_bs32.py
    ${BASE_DIR}/rsnet/rsnet_local_cutmix_sz224_bs32.py
    ${BASE_DIR}/rsnet/rsnet_local_saliencymix_sz224_bs32.py
    ${BASE_DIR}/rsnet/rsnet_local_starmix_sz224_bs32.py
    ${BASE_DIR}/rsnet/rsnet_local_starmixplus_sz224_bs32.py
)


GPUS=$1
PY_ARGS=${@:2}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

for CFG in "${CFG_LIST[@]}"; do
    WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
    $PYTHON -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        tools/train.py $CFG --work_dir $WORK_DIR \
        --seed 0 --launcher pytorch ${PY_ARGS}
done
