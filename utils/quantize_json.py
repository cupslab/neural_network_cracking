#!/usr/bin/env python

import sys
import argparse
import json
import decimal as decimal

max_value = -1
min_value = 1
min_non_zero = 1
previous = 0

fp_max = 5
fp_min = -5

def quantize(bits, fixed_point):
    size = '1.' + ('0' * bits)
    fp_bits = 2**bits
    fp_domain = (fp_max + abs(fp_min))
    fp_offset = 0 # fp_domain / 2
    fp_factor = fp_bits / fp_domain
    def quantize(x):
        global max_value, min_value, previous, min_non_zero
        answer = decimal.Decimal(x)
        max_value = max(answer, max_value)
        min_value = min(answer, min_value)
        if answer != 0:
            min_non_zero = min(min_non_zero, abs(answer))
        if fixed_point:
            temp = int((float(answer) + fp_offset) * fp_factor)
        else:
            temp = float(answer.quantize(decimal.Decimal(size)))
        previous = temp
        return temp
    return quantize

def main(args):
    json.dump(json.load(
        args.ifile, parse_float=quantize(args.bits, args.fixed_point)),
              args.ofile)
    sys.stderr.write(', '.join(map(str, [
        max_value, min_value, min_non_zero])) + '\n')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--ifile', type=argparse.FileType('r'),
                        help='Input file. Default is stdin. ',
                        default=sys.stdin)
    parser.add_argument('-o', '--ofile', type=argparse.FileType('w'),
                        help='Output file. Default is stdout. ',
                        default=sys.stdout)
    parser.add_argument('-b', '--bits', default=4, type=int)
    parser.add_argument('-f', '--fixed-point', action='store_true')
    main(parser.parse_args())
