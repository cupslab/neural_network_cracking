#!/usr/bin/env bash

WORKDIR="experiments/constant"
python3 generate_test_data.py --line-count 10000 \
        --ofile $WORKDIR/input.txt constant
python3 pwd_guess.py --arch-file $WORKDIR/constant_arch.json \
        --weight-file $WORKDIR/constant_weight.h5 \
        --config $WORKDIR/constant_config.json \
        --log-file $WORKDIR/log.txt \
        --pwd-file $WORKDIR/input.txt
