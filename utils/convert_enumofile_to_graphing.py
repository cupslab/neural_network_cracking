#!/usr/bin/env python
import argparse
import os.path
def create_output(args):
    prob = "0x0.1p-1"
    max_guess = 0
    with open(args.inp_file,"r") as inp, open(os.path.join(args.op_dir,"lookupresults."+args.condition),"w") as out:
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
            out.write("\t".join(op_cols)+"\n")
            if guess_no > max_guess:
                max_guess = guess_no
    with open(os.path.join(args.op_dir,"totalcounts."+args.condition),"w") as out:
        out.write(args.condition+":Total count\t"+str(max_guess)+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Convert the output of guesser enum_ofile to the input required by graphing Rscript"))
    parser.add_argument('inp_file')
    parser.add_argument('op_dir')
    parser.add_argument('--user',default="no_name")
    parser.add_argument('--condition', default="4c8")
    parser.add_argument('--guess_type', default="WRGOMI")
    args = parser.parse_args()
    create_output(args)