#!/usr/env/bin python3
import argparse
import os
import os.path
import shutil
import subprocess
import json
def parse_args():
    p = argparse.ArgumentParser(description="Script to automate a training and guess runs with specified config files" )
    p.add_argument("--rundir",default=".", help="Specify the directory from where the run will be launched"
                                                " and the output files will be populated")
    p.add_argument("--train-config", required=True,help = "The path of the training config-args file. Specify both "
                                                          "the config and args here. Paths should be relative to rundir")
    p.add_argument("--train-only",action="store_true")
    p.add_argument("--guess-config",help="Path of the guessing config-args file. Specify both the config and args here."
                                         "Paths should be relative to rundir")
    p.add_argument("--reason", help="Some helpful text to understand why this run was launched")
    return p.parse_args()
def validate_args(args):
    if not os.path.exists(args.train_config):
        raise FileNotFoundError("Train file not found at {}".format(args.train_config))
    if not args.train_only:
        if not os.path.exists(args.guess_config):
            raise FileNotFoundError("Guessing file not found at {}".format(args.guess_config))
    if args.train_only and args.guess_config:
        print("Guess config will be ignored as the run is train only")

if __name__ == "__main__":
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    pwd_guess = os.path.join(root_dir, "pwd_guess.py")
    convert_py = os.path.join(root_dir, "utils","convert_enumofile_to_graphing.py")
    try:
        os.mkdir(args.rundir)
    except FileExistsError:
        print("Path already exists, will use the existing directory")

    validate_args(args)
    try:
        shutil.copy(args.train_config, args.rundir)
    except shutil.SameFileError:
        pass
    if not args.train_only:
        try:
            shutil.copy(args.guess_config, args.rundir)
        except shutil.SameFileError:
            pass
    os.chdir(args.rundir)
    if args.reason:
        with open("reason.txt","w") as readme:
            readme.write(args.reason)
    if not os.path.exists(pwd_guess):
        raise FileNotFoundError("Couldn't find the pwd_guess.py script at {}".format(pwd_guess))

    #Make training attempt
    train_cmd = "python3 {} --config-args \"{}\" |& tee training.log".format(pwd_guess, os.path.basename(args.train_config))
    print(train_cmd)
    ret = subprocess.call(train_cmd, shell=True, executable='/bin/bash')
    if ret != 0:
        raise RuntimeError("The training process returned non-zero error code. Look in training.log for more information")
    if args.train_only:
        print("Completed training, exiting")
        exit(0)

    #Make guessing attempt
    guess_cmd = "python3 {} --config-args \"{}\" |& tee guess.log".format(pwd_guess, os.path.basename(args.guess_config))
    print(guess_cmd)
    ret = subprocess.call(guess_cmd, shell=True,executable='/bin/bash')
    if ret != 0:
        raise RuntimeError("The guessing process returned non-zero error code. Look in guess.log for more information")
    guess_conf = json.load(open(os.path.basename(args.guess_config),"r"))

    #Convert to format for plotting
    guess_op_file = guess_conf['args']['enumerate_ofile']
    convert_cmd = "python3 {} \"{}\" . --condition {}".format(convert_py, guess_op_file, os.path.basename(os.getcwd()))
    ret = subprocess.call(convert_cmd, shell= True, executable='/bin/bash')
    if ret != 0:
        raise RuntimeError("Convert command failed with the commandline {}".format(convert_cmd))
    modplot_path = os.path.join(root_dir,"graphing","ModPlotResults.R")
    plot_cmd = "Rscript {} makeplot plot.pdf".format(modplot_path)

    ret = subprocess.call(plot_cmd, shell=True, executable='/bin/bash')
    if ret !=0:
        raise RuntimeError("Plotting failed with command {}".format(plot_cmd))

    print("DONE!!")
    exit(0)