#!/usr/bin/env bash
# EXAMPLE:
# `CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./train.sh ${CONFIG_FILE} 4`

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/main.py $CONFIG --launcher pytorch ${@:3}