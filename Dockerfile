# Based on:
# https://www.password-guessing.org/blog/post/cupslab-neural-network-cracking-manual/

# by default use graphics card configuration
# overide using --build-arg in `docker build` command
ARG THEANO_CONFIG=.theanorc.gpu
ARG BASE_IMAGE=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

FROM $BASE_IMAGE
MAINTAINER Josh Tan <jstan@cs.cmu.edu>

RUN apt-get update && apt-get install -y \
  build-essential \
  g++ \
  git \
  libblas-dev \
  python-setuptools \
  python3 \
  python3-dev \
  python3-pip \
  unzip \
  wget

RUN mkdir /nn
WORKDIR /nn

ADD requirements.txt \
    markov_model.py \
    markov_model_tests.py \
    parallel_generate_markov.sh \
    pwd_guess.py \
    pwd_guess_ctypes.pyx \
    pwd_guess_unit.py \
    setup.py \
    /nn/

# first upgrade pip to support -trusted-host param
RUN pip3 install -Iv pip==10.0.1
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

RUN wget https://github.com/Theano/Theano/archive/rel-0.8.2.zip && \
    unzip rel-0.8.2.zip && \
    cd Theano-rel-0.8.2 && \
    python3 setup.py develop --user

ADD $THEANO_CONFIG /root/.theanorc

RUN git clone https://github.com/fchollet/keras.git && \
    cd keras && \
    git checkout 24b5e80667c8998d7e5e9689085fecc92a9506d3 && \
    python3 setup.py install

RUN git clone https://github.com/EderSantana/seya.git && \
    cd seya && \
    git checkout 49d1bfd66f7442b7ce09dc86f7e78a32b132dd48 && \
    python3 setup.py install

RUN python3 setup.py build_ext --inplace
