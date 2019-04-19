#!/usr/bin/env python3

import argparse
import os
import subprocess


CPU_TAG_NAME = "pwd-nn-cpu"
GPU_TAG_NAME = "pwd-nn-gpu"


def main(args):

    if args.action == "build-cpu" or args.action == "build-gpu":

        if args.action == "build-cpu":
            build_args = ["--build-arg", "BASE_IMAGE=tensorflow/tensorflow:1.4.0-py3",
                          "--build-arg", "REQUIREMENTS=requirements-tensorflow-cpu-1.4.txt"]
            tag_name = CPU_TAG_NAME

        elif args.action == "build-gpu":
            build_args = ["--build-arg", "BASE_IMAGE=tensorflow/tensorflow:1.4.0-gpu-py3",
                          "--build-arg", "REQUIREMENTS=requirements-tensorflow-1.4.txt"]
            tag_name = GPU_TAG_NAME

        subprocess.check_call(["docker", "build", "--tag", "cupslab/{}".format(tag_name)] +
                              build_args + ["."])

    else:

        # volume map configs and pre-built networks
        configs_dir = os.path.join(os.getcwd(), "configs")
        embedding_dir = os.path.join(os.getcwd(), "embedding")
        graphing_dir = os.path.join(os.getcwd(), "graphing")
        js_dir = os.path.join(os.getcwd(), "js")
        measurement_dir = os.path.join(os.getcwd(), "measurement")
        pre_built_dir = os.path.join(os.getcwd(), "pre_built_networks")
        strategy_simulation_dir = os.path.join(os.getcwd(), "strategy_simulation")
        test_data_dir = os.path.join(os.getcwd(), "test_data")
        utils_dir = os.path.join(os.getcwd(), "utils")
        git_dir = os.path.join(os.getcwd(), ".git")

        mappings = []        
        mappings.extend(["-v", "{}:{}".format(configs_dir, "/nn/configs")])
        mappings.extend(["-v", "{}:{}".format(embedding_dir, "/nn/embedding")])
        mappings.extend(["-v", "{}:{}".format(graphing_dir, "/nn/graphing")])
        mappings.extend(["-v", "{}:{}".format(js_dir, "/nn/js")])
        mappings.extend(["-v", "{}:{}".format(measurement_dir, "/nn/measurement")])
        mappings.extend(["-v", "{}:{}".format(pre_built_dir, "/nn/pre_built_networks")])
        mappings.extend(["-v", "{}:{}".format(strategy_simulation_dir, "/nn/strategy_simulation")])
        mappings.extend(["-v", "{}:{}".format(test_data_dir, "/nn/test_data")])
        mappings.extend(["-v", "{}:{}".format(utils_dir, "/nn/utils")])
        mappings.extend(["-v", "{}:{}".format(git_dir, "/nn/.git")])

        if args.output_dir:
            source = os.path.join(os.getcwd(), args.output_dir)
            dest = "/nn/output"
            mappings.extend(["-v", "{}:{}".format(source, dest)])

        if args.input_dir:
            source = os.path.join(os.getcwd(), args.input_dir)
            dest = "/nn/input"
            mappings.extend(["-v", "{}:{}".format(source, dest)])

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
                        help="Relative directory containing password sets",
                        default="nn_input")                        
    parser.add_argument("--output_dir",
                        help="Relative directory to store ouput files in",
                        default="nn_output")
    
    args = parser.parse_args()
    main(args)
