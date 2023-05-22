#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29502}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p basemodel -x SH-IDC1-10-142-5-16 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=112 --quotatype=spot --async \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch ${@:3}
