#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

BASE_DIR="/yuchang/lsy_jx/AGVBench/configs/classification/_metaformer/tju600"

# identityformer_s24
# randformer_s24
# poolformerv2_s24
# convformer_s18
# caformer_s18
# identityformer_s12   ← 轻量级对照
CFG_LIST=(
    # ${BASE_DIR}/identityformer_s12_mixup_bs32_sz224.py
    # ${BASE_DIR}/identityformer_s24_mixup_bs32_sz224.py
    # ${BASE_DIR}/identityformer_s36_mixup_bs32_sz224.py
    # ${BASE_DIR}/randformer_s12_mixup_bs32_sz224.py
    # ${BASE_DIR}/randformer_s24_mixup_bs32_sz224.py
    # ${BASE_DIR}/randformer_s36_mixup_bs32_sz224.py
    ${BASE_DIR}/poolformerv2_s12_mixup_bs32_sz224.py
    ${BASE_DIR}/poolformerv2_s24_mixup_bs32_sz224.py
    ${BASE_DIR}/poolformerv2_s36_mixup_bs32_sz224.py
    ${BASE_DIR}/convformer_s18_mixup_bs32_sz224.py
    ${BASE_DIR}/convformer_s36_mixup_bs32_sz224.py
    ${BASE_DIR}/caformer_s18_mixup_bs32_sz224.py
    ${BASE_DIR}/caformer_s36_mixup_bs32_sz224.py
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
