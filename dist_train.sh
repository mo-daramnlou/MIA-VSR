#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-4321}

# usage
if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "bash dist_train.sh [number of gpu] [path to option file]"
    exit
fi

# The original PYTHONPATH was incorrect for a script in the root directory.
# It should point to the project root, which is the current directory '.'.
# We also switch to `torchrun`, the modern replacement for `torch.distributed.launch`,
# to better handle distributed training arguments like --local-rank.
PYTHONPATH=".:${PYTHONPATH}" \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    recurrent_mix_precision_train.py -opt $CONFIG --launcher pytorch ${@:3}
