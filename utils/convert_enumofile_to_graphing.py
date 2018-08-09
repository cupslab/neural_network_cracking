#!/usr/bin/env python3
import argparse
import os.path
from collections import Counter
import numpy as np

def create_output(args):
    prob = "0x0.1p-1"
    max_guess = 0
    if args.deduplicate:
        with open(args.deduplicate,"r") as f:
            deduplicated = [x.rstrip("\n") for x in f.readlines()]
        counts = Counter(deduplicated)
    probs = []
    guess_list = []
    with open(args.inp_file, "r") as inp, open(os.path.join(args.op_dir, "lookupresults."+args.condition), "w") as out:
        for line in inp:
            cols = line.split('\t')
            op_cols=[]
            op_cols.append(args.user)
            op_cols.append(args.condition)
            op_cols.append(cols[0])
            op_cols.append(prob)
            op_cols.append("")
            guess_no = int(round(float(cols[2])))
            op_cols.append(str(guess_no))
            op_cols.append(args.guess_type)
            if args.deduplicate:
                for _ in range(counts[cols[0]]):
                    out.write("\t".join(op_cols)+"\n")
                    probs.append(float(cols[1]))
                    guess_list.append(guess_no)
            else:
                out.write("\t".join(op_cols)+"\n")
                probs.append(float(cols[1]))
                guess_list.append(guess_no)
            if guess_no > max_guess:
                max_guess = guess_no
    with open(os.path.join(args.op_dir,"totalcounts."+args.condition),"w") as out:
        out.write(args.condition+":Total count\t"+str(max_guess)+"\n")

    perplexity = calc_perplexity(probs)
    sampling_at_guess_num = [1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24]
    percentile = fraction_guessed(guess_list, sampling_at_guess_num)

    out = "Perplexity : {} Entropy: {:.6f}\nGuess number -> Fraction of passwords guessed\n".format(
        perplexity, np.log2(perplexity))
    for guess_num in sampling_at_guess_num:
        out += "{:.0E} -> {:.2f}%\n".format(guess_num, percentile[guess_num])
    with open(os.path.join(args.op_dir, "performance_results.{}".
            format(args.condition)), "w") as out_file:
        out_file.write(out)

def calc_perplexity(probs):
    return 2 ** (-(1/len(probs))*np.sum(np.log2(probs)))

def fraction_guessed(guess_list,guess_sampling):
    guess_list = sorted(guess_list)
    figured_idx = {guess: 0 for guess in guess_sampling}

    for idx, val in enumerate(guess_list):
        for guess in guess_sampling:
            if val < guess:
                figured_idx[guess] = idx
    for idx in figured_idx:
        figured_idx[idx] = 100*figured_idx[idx]/len(guess_list)
    return figured_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Convert the output of guesser enum_ofile to the input required by graphing Rscript"))
    parser.add_argument('inp_file')
    parser.add_argument('op_dir')
    parser.add_argument('--user',default="no_name")
    parser.add_argument('--condition', default="4c8")
    parser.add_argument('--guess_type', default="WRGOMI")
    parser.add_argument('--deduplicate', help="The test file used to generate guessing numbers", required=True)
    args = parser.parse_args()
    create_output(args)