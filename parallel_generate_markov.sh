#!/usr/bin/env bash

NUM=$1
MODEL_FILE=$2
CONFIG_FILE=$3
ORDER=$4
SMOOTHING=$5
CORES=$6
WORKDIR=$(gmktemp -d --tmpdir=$(pwd) -t markov-parallel-workXXXXXXXX)
echo "Using $WORKDIR..."
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
time seq 1 $CORES | parallel --results "$WORKDIR" \
                             --jobs "$CORES" \
                             --arg-file - \
                             "python3 $DIR/markov_model.py --model-file '$MODEL_FILE' --config '$CONFIG_FILE' --k-order $ORDER --smoothing $SMOOTHING --config-values 'random_walk_seed_num=$NUM' --ofile $WORKDIR/{}-output" &> /dev/null
