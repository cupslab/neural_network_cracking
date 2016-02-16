#!/usr/bin/env python

import sys
import argparse
import csv
import os

def main(args):
    pwds = list([line.rstrip(os.linesep) for line in args.password_list])
    ofile = open(os.path.join(
        args.output_dir, 'lookupresults.' + args.name), 'w')
    writer = csv.writer(ofile, delimiter='\t', quotechar=None)
    max_gn = 0
    guess_numbers = {}
    for row in csv.reader(args.guess_numbers, delimiter='\t', quotechar=None):
        if len(row) == 6:
            pwd, prob_str, guess_number, var, num, confidence = row
        elif len(row) == 2:
            pwd, guess_number = row
            prob_str, var, num, confidence = '0.1337', 0, 1, 0
        else:
            sys.stderr.write('Error, expected 2 or 6 rows and found %d %s\n' %
                             (len(row), str(row)))
            continue
        if pwd not in pwds:
            continue
        try:
            guess_number_round = int(round(float(guess_number), 0))
        except ValueError:
            try:
                guess_number_round = int(guess_number)
            except ValueError as e:
                sys.stderr.write(
                    'Error: Cannot interpret guess number "%s"\n',
                    guess_number)
                continue
        except OverflowError as e:
            guess_number_round = -1
        guess_numbers[pwd] = [float.hex(float(prob_str)), guess_number_round]
        max_gn = max(max_gn, guess_number_round)
    for pwd in pwds:
        if pwd in guess_numbers:
            writer.writerow(['no_user', args.name, pwd,
                             guess_numbers[pwd][0], '0x0.1p-1',
                             guess_numbers[pwd][1], 'WRGOMI'])
        else:
            writer.writerow(['no_user', args.name, pwd,
                             -1, '0x0.1p-1',
                             -1, 'WRGOMI'])
    ofile.close()
    with open(os.path.join(
            args.output_dir, 'totalcounts.' + args.name), 'w') as totalcfile:
        totalcfile.write(args.name + ':Total count\t' + str(max_gn) + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='convert to graphing format')
    parser.add_argument('password_list', type = argparse.FileType('r'))
    parser.add_argument('guess_numbers', type = argparse.FileType('r'))
    parser.add_argument('name')
    parser.add_argument('-o', '--output-dir', default='./')
    main(parser.parse_args())
