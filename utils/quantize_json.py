#!/usr/bin/env python

import sys
import argparse
import json
import decimal as decimal

max_value = -1
min_value = 1
previous = 0

def quantize(bits):
    size = '1.' + ('0' * bits)
    def quantize(x):
        global max_value, min_value, previous
        answer = decimal.Decimal(x)
        max_value = max(answer, max_value)
        min_value = min(answer, min_value)
        temp = answer # - previous
        previous = answer
        return float(temp.quantize(decimal.Decimal(size)))
    return quantize

def main(args):
    json.dump(json.load(
        args.ifile, parse_float=quantize(args.bits)), args.ofile)
    sys.stderr.write(str(max_value) + ', ' + str(min_value) + '\n')

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
