#!/usr/bin/env bash

NAME="$1"
WORKDIR="experiments/$NAME"
mkdir -p $WORKDIR

python3 generate_test_data.py --line-count 100000 \
        --ofile $WORKDIR/input.txt "$NAME"

python3 pwd_guess.py --arch-file $WORKDIR/"$NAME"_arch.json \
        --weight-file $WORKDIR/"$NAME"_weight.h5 \
        --config configs/"$NAME"_config.json \
        --log-file $WORKDIR/log.txt \
        --pwd-file $WORKDIR/input.txt \
        --enumerate-ofile $WORKDIR/guesses.txt

sort -g -k2 -r -s -t$'\t' $WORKDIR/guesses.txt > $WORKDIR/sorted_guesses.txt
