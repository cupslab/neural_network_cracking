#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser(description="Create graphs in the current directory")
parser.add_argument("outfile")
parser.add_argument("--remote", action="store_true", help="Use this option on a remote"
                                                          "ssh machine.")
args = parser.parse_args()
import matplotlib
if args.remote:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
if __name__ == "__main__":

    fnames = glob.glob("lookupresults.*")
    assert len(fnames) != 0, "Couldn't find file of the name lookupresults.* in current directory"
    conditions = set()
    for f in fnames:
        m = re.search(r".*?\.(.*)", f)
        conditions.add(m.group(1))

    for condition in conditions:
        guesses = []
        with open("lookupresults.{}".format(condition)) as f:
            for line in f:
                try:
                    guess_num = int(line.split("\t")[5])
                    if guess_num < 0:
                        continue
                    guesses.append(guess_num)
                except:
                    raise ValueError("Error while parsing the following line.\
                    Expected integer at column 6\n{}".format(line))
            guesses.sort()
            counts = np.arange(1, len(guesses)+1)
            counts = (counts/len(counts))*100
            plt.step(guesses, counts, label=condition)
    plt.xscale('log')
    axes = plt.gca()
    axes.set_xlim([1, 10e25])
    plt.legend()
    plt.grid(True)
    plt.xlabel("Guesses")
    plt.ylabel("Percent guessed")
    plt.savefig(args.outfile)
