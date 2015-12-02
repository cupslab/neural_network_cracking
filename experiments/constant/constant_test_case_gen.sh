#!/usr/bin/env bash

WORKDIR="experiments/constant"
python3 pwd_guess.py --arch-file $WORKDIR/constant_arch.json \
        --weight-file $WORKDIR/constant_weight.h5 \
        --enumerate-ofile $WORKDIR/guesses.txt \
        --log-file $WORKDIR/log_guess.txt \
        --config $WORKDIR/constant_config.json

sort -g -k2 -r -s -t$'\t' $WORKDIR/guesses.txt > sorted_guesses.txt
