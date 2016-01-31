#!/usr/bin/env python

import sys
import argparse
import csv
import os

def main(args):
    prev = None
    ctr = 0
    fn = (lambda x: float.hex(float(x))) if args.hex else str
    for pwd in args.ifile:
        pwd = pwd.rstrip(os.linesep)
        if pwd == prev:
            ctr += 1
        else:
            if ctr != 0:
                args.ofile.write('%s\t%s\n' % (pwd, fn(ctr)))
            prev = pwd
            ctr = 1

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Sorted file to word freq format. ')
    parser.add_argument('-i', '--ifile', type = argparse.FileType('r'),
                        help = 'Input file. Default is stdin. ',
                        default = sys.stdin)
    parser.add_argument('-o', '--ofile', type = argparse.FileType('w'),
                        help = 'Output file. Default is stdout. ',
                        default = sys.stdout)
    parser.add_argument('--hex', action='store_true')
    main(parser.parse_args())
