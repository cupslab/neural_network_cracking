import sys
import argparse
import csv
import os

def main(args):
    with open(args.guess_number_file, 'r') as gnfile:
        guess_numbers = dict([
            (row[0], float(row[2]))
            for row in csv.reader(gnfile, delimiter='\t', quotechar=None)])
    with open(args.training_file, 'r') as filter_file:
        for rank, line in enumerate(filter_file):
            pwd = line.strip(os.linesep)
            if (pwd not in guess_numbers or
                not ((guess_numbers[pwd] * (10**args.tolerance) >= rank) and
                     (guess_numbers[pwd] / (10**args.tolerance) <= rank))):
                sys.stdout.write('%s\t%d\n' % (pwd, rank))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_file')
    parser.add_argument('guess_number_file')
    parser.add_argument('-t', '--tolerance', type=float, default=2,
                        help=('10 based exponent for tolerance ratio. '
                              '2 means filter out passwords that are '
                              'not with a factor of 10^2. Default is 2. '))
    main(parser.parse_args())
