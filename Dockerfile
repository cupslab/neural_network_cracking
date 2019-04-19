ARG BASE_IMAGE=tensorflow/1.4.0-gpu-py3
FROM $BASE_IMAGE

MAINTAINER Josh Tan <jstan@cs.cmu.edu>

RUN apt-get update && apt-get install -y git

RUN mkdir /nn
WORKDIR /nn

ARG REQUIREMENTS=requirements-tensorflow-1.4.txt
COPY $REQUIREMENTS /nn/

ADD markov_model.py \
    markov_model_tests.py \
    parallel_generate_markov.sh \
    pwd_guess.py \
    pwd_guess_ctypes.pyx \
    pwd_guess_unit.py \
    pwd_wrapper.py \
    pylintrc \
    setup.py \
    /nn/

ARG REQUIREMENTS
RUN pip3 install -r /nn/$REQUIREMENTS

RUN python3 setup.py build_ext --inplace
