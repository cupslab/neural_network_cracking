#!/usr/bin/env python

import sys
import argparse
import json
import decimal as decimal

class DeltaCoder(object):
    def __init__(self):
        self.prev = 0

    def __call__(self, x):
        answer = x - self.prev
        self.prev = x
        return answer

    def curry(self, fn):
        return lambda y: self(fn(y))

def quantize(bits, fixed_point, delta):
    size = '1.' + ('0' * bits)
    def quantize(x):
        return float(decimal.Decimal(x).quantize(decimal.Decimal(size)))
    def fixed_point_q(x):
        return int(float(x) * fixed_point)
    if fixed_point == -1:
        answer_fn = quantize
    else:
        answer_fn = fixed_point_q
    if delta:
        return DeltaCoder().curry(answer_fn)
    return answer_fn

def main(args):
    oobj = json.load(args.ifile, parse_float=quantize(
        args.bits, args.fixed_point, args.delta_coding))
    if args.remove_spaces:
        str_value = json.dumps(oobj)
        args.ofile.write(str_value.replace(' ', ''))
    else:
        json.dump(oobj, args.ofile)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Compress json data by quantizing floats')
    parser.add_argument('-i', '--ifile', type=argparse.FileType('r'),
                        help='Input file. Default is stdin. ',
                        default=sys.stdin)
    parser.add_argument('-o', '--ofile', type=argparse.FileType('w'),
                        help='Output file. Default is stdout. ',
                        default=sys.stdout)
    parser.add_argument('-b', '--bits', default=4, type=int)
    parser.add_argument('-f', '--fixed-point', type=int, default=-1,
                        help='Factor to multiply data by')
    parser.add_argument('-d', '--delta-coding', action='store_true')
    parser.add_argument('-r', '--remove-spaces', action='store_true')
    main(parser.parse_args())
