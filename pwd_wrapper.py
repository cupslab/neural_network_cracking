#!/usr/env/bin python3
import argparse
import os
import os.path
import shutil
import subprocess
import json
root_dir = os.path.dirname(os.path.abspath(__file__))
def parse_args():
    p = argparse.ArgumentParser(description="Script to automate a training and guess runs with specified config files" )
    p.add_argument("--rundir", default=".", help="Specify the directory from where the run will be launched"
                                                " and the output files will be populated")
    p.add_argument("--train-config", help = "The path of the training config-args file. Specify both "
                                                          "the config and args here. Paths should be relative to rundir")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--start-secondary", action="store_true", help="To skip initial training and start with "
                                                                  "secondary training")
    p.add_argument("--secondary-config", help="The path of the secondary training config-args file. Paths "
                                              "relative to rundir")
    p.add_argument("--start-guessing", action="store_true", help="To just proceed with guessing run")
    p.add_argument("--guess-config", help="Path of the guessing config-args file. Specify both the config and args here. "
                                         "Paths should be relative to rundir")
    p.add_argument("--reason", help="Some helpful text to understand why this run was launched")
    return p.parse_args()


def validate_args(args):
    if not args.start_guessing and not args.start_secondary and not os.path.exists(args.train_config):
        raise FileNotFoundError("Train file not found at {}".format(args.train_config))
    if not args.train_only:
        if not os.path.exists(args.guess_config):
            raise FileNotFoundError("Guessing file not found at {}".format(args.guess_config))

        if args.secondary_config and not os.path.exists(args.secondary_config):
            raise FileNotFoundError("Secondary config-args not found at {}".format(args.secondary_config))
    if args.train_only and args.guess_config:
        print("Guess config will be ignored as the run is train only")

def get_current_commit():
    try:
        commit_run = subprocess.run("git rev-parse HEAD", shell=True, check=True,
                                    stdout=subprocess.PIPE, universal_newlines=True)
        commit_hash = commit_run.stdout.strip()
        return commit_hash
    except subprocess.CalledProcessError:
        return None


if __name__ == "__main__":
    args = parse_args()
    pwd_guess = os.path.join(root_dir, "pwd_guess.py")
    convert_py = os.path.join(root_dir, "utils", "convert_enumofile_to_graphing.py")
    try:
        os.mkdir(args.rundir)
    except FileExistsError:
        print("Path already exists, will use the existing directory")

    validate_args(args)
    os.chdir(args.rundir)
    commit = get_current_commit()
    if args.reason:
        with open("reason.txt", "w") as readme:
            readme.write(args.reason)
    if commit:
        with open("commit.hash", "w") as commit_file:
            commit_file.write(commit)
    if not os.path.exists(pwd_guess):
        raise FileNotFoundError("Couldn't find the pwd_guess.py script at {}".format(pwd_guess))
    statvfs = os.statvfs(args.rundir)
    free_bytes = statvfs.f_frsize * statvfs.f_bfree
    if free_bytes < 200000000:
        raise OSError("Minimum disk space of 500MB required to reliably run this code")

    #Make training attempt
    if not args.start_secondary and not args.start_guessing:
        try:
            shutil.copy(args.train_config, args.rundir)
        except shutil.SameFileError:
            pass

        train_cmd = "python3 {} --config-args \"{}\" |& tee training.log".format(pwd_guess, os.path.basename(args.train_config))
        print(train_cmd)
        ret = subprocess.call(train_cmd, shell=True, executable='/bin/bash')
        train_conf = json.load(open(os.path.basename(args.train_config), "r"))
        try:
            shutil.copy(train_conf['args']['weight_file'],
                        os.path.join(args.rundir, train_conf['args']['weight_file'])+".orig")
        except shutil.SameFileError:
            pass
        if ret != 0:
            raise RuntimeError("The training process returned non-zero error code. Look in training.log for more information")
        if args.train_only:
            print("Completed training, exiting")
            exit(0)

    #Make secondary training attempt
    if args.secondary_config and not args.start_guessing:
        secondary_cmd = "python3 {} --config-args \"{}\" |& tee secondary.log".\
            format(pwd_guess, os.path.basename(args.secondary_config))
        print(secondary_cmd)
        ret = subprocess.call(secondary_cmd, shell=True, executable='/bin/bash')
        if ret != 0:
            raise RuntimeError("The secondary training process returned non-zero error code."
                               "Look in the secondary.log for more information")


    #Make guessing attempt
    try:
        shutil.copy(args.guess_config, args.rundir)
    except shutil.SameFileError:
        pass

    guess_cmd = "python3 {} --config-args \"{}\" |& tee guess.log".format(pwd_guess, os.path.basename(args.guess_config))
    print(guess_cmd)
    ret = subprocess.call(guess_cmd, shell=True,executable='/bin/bash')
    if ret != 0:
        raise RuntimeError("The guessing process returned non-zero error code. Look in guess.log for more information")
    guess_conf = json.load(open(os.path.basename(args.guess_config), "r"))

    #Convert to format for plotting
    guess_op_file = guess_conf['args']['enumerate_ofile']
    test_fname = guess_conf['config']['password_test_fname']
    convert_cmd = "python3 {} \"{}\" . --condition {} --deduplicate {}".format(
        convert_py, guess_op_file, os.path.basename(os.getcwd()), test_fname)
    ret = subprocess.call(convert_cmd, shell=True, executable='/bin/bash')
    if ret != 0:
        raise RuntimeError("Convert command failed with the commandline {}".format(convert_cmd))
    grapher_path = os.path.join(root_dir, "graphing", "grapher.py")
    plot_cmd = "python3 {} plot.png --remote".format(grapher_path)

    ret = subprocess.call(plot_cmd, shell=True, executable='/bin/bash')
    if ret != 0:
        raise RuntimeError("Plotting failed with command {}".format(plot_cmd))

    print("SUCCESSFULLY COMPLETED THE WHOLE PROCESS!!")
    exit(0)