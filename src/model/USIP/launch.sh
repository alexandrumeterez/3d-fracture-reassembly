#!/bin/bash
if [[ $# -ne 3 ]]; then
    echo "Illegal number of parameters" >&2
    exit 2
fi

LR=$1
DATA_DIR=$2
CKPT_DIR=$3

bsub -W 10:00 -R rusage[ngpus_excl_p=1,mem=30000] python customnet/train_detector.py --lr $LR --checkpoints_dir $CKPT_DIR --dataroot $DATA_DIR
