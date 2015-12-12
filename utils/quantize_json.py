#!/usr/bin/env python

import sys
import argparse
import json
import decimal as decimal

def quantize(bits):
    size = '1.' + ('0' * bits)
    return lambda x: float(decimal.Decimal(x).quantize(decimal.Decimal(size)))

def main(args):
    json.dump(json.load(
        args.ifile, parse_float=quantize(args.bits)), args.ofile)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--ifile', type=argparse.FileType('r'),
                        help='Input file. Default is stdin. ',
                        default=sys.stdin)
    parser.add_argument('-o', '--ofile', type=argparse.FileType('w'),
                        help='Output file. Default is stdout. ',
                        default=sys.stdout)
    parser.add_argument('-b', '--bits', default=4, type=int)
    main(parser.parse_args())
