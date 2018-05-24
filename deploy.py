#!/usr/bin/env python3

import argparse
import os
import subprocess


CPU_TAG_NAME = "pwd-nn-cpu"
GPU_TAG_NAME = "pwd-nn-gpu"


def main(args):

    if args.action == "build-cpu" or args.action == "build-gpu":

        if args.action == "build-cpu":
            build_args = ["--build-arg", "BASE_IMAGE=ubuntu:16.04",
                          "--build-arg", "THEANO_CONFIG=.theanorc.cpu"]
            tag_name = CPU_TAG_NAME

        elif args.action == "build-gpu":
            build_args = ["--build-arg", "BASE_IMAGE=nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04",
                          "--build-arg", "THEANO_CONFIG=.theanorc.gpu"]
            tag_name = GPU_TAG_NAME

        subprocess.check_call(["docker", "build", "--tag", "cupslab/{}".format(tag_name)] +
                              build_args + ["."])

    else:

        # volume map configs and pre-built networks
        configs_dir = os.path.join(os.getcwd(), "configs")
        pre_built_dir = os.path.join(os.getcwd(), "pre_built_networks")

        mappings = []        
        mappings.extend(["-v", "{}:{}".format(configs_dir, "/nn/configs")])
        mappings.extend(["-v", "{}:{}".format(pre_built_dir, "/nn/pre_built_networks")])

        if args.output_dir:
            source = os.path.join(os.getcwd(), args.output_dir)
            dest = "/nn/output"
            mappings.extend(["-v", "{}:{}".format(source, dest)])

        if args.input_dir:
            source = os.path.join(os.getcwd(), args.input_dir)
            dest = "/nn/input"
            mappings.extend(["-v", "{}:{}".format(source, dest)])

        if args.develop:
            mappings.extend(["-v", "{}:{}".format(os.getcwd(), "/nn/")])

        if args.action == "run-cpu":
            runtime_args = []
            tag_name = CPU_TAG_NAME
            
        else: # args.action == "run-gpu"
            runtime_args = ["--runtime", "nvidia"]
            tag_name = GPU_TAG_NAME

        # instantiate an interactive container running bash
        subprocess.call(["docker", "run", "-it"] + mappings + runtime_args +
                        ["cupslab/{}:latest".format(tag_name), "/bin/bash"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy neural network password guesser")
    parser.add_argument("action",
                        choices=["build-cpu", "build-gpu", "run-cpu", "run-gpu"])
    parser.add_argument("--input_dir",
                        help="Relative directory containing password sets")
    parser.add_argument("--output_dir",
                        help="Relative directory to store ouput files in")
    parser.add_argument("--develop",
                        action="store_true",
                        help="Development mode. Mirrors repo contents within the container.")
    
    args = parser.parse_args()
    main(args)
