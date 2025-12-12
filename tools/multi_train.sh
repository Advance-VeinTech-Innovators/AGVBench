#!/usr/bin/env bash
#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG_LIST=(
# /lisiyuan/jx/AGVBench/configs/classification/settings/ampvnet/lr_1e-3/autoaug_sz224_bs32.py
/lisiyuan/jx/AGVBench/configs/classification/settings/ampvnet/lr_1e-3/mixups_sz224_bs32.py
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
