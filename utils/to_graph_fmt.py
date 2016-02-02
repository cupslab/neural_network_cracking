#!/usr/bin/env python

import sys
import argparse
import csv
import os

def main(args):
    pwds = set([line.strip(os.linesep) for line in args.password_list])
    ofile = open(os.path.join(
        args.output_dir, 'lookupresults.' + args.name), 'w')
    writer = csv.writer(ofile, delimiter='\t', quotechar=None)
    max_gn = 0
    for row in csv.reader(args.monte_carlo, delimiter='\t', quotechar=None):
        pwd, prob_str, guess_number, var, num, confidence = row
        if pwd in pwds:
            guess_number_round = int(round(float(guess_number), 0))
            writer.writerow(['no_user', args.name, pwd,
                             float.hex(float(prob_str)), '0x0.1p-1',
                             guess_number_round, 'WRGOMI'])
            max_gn = max(max_gn, guess_number_round)
    ofile.close()
    with open(os.path.join(
            args.output_dir, 'totalcounts.' + args.name), 'w') as totalcfile:
        totalcfile.write(args.name + ':Total count\t' + str(max_gn) + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='convert to graphing format')
    parser.add_argument('password_list', type = argparse.FileType('r'))
    parser.add_argument('monte_carlo', type = argparse.FileType('r'))
    parser.add_argument('name')
    parser.add_argument('-o', '--output-dir', default='./')
    main(parser.parse_args())
