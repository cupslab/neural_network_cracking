#!/usr/bin/env python

import sys
import argparse
import json
import decimal as decimal

class Filter(object):
    def curry(self, fn):
        return lambda y: self(fn(y))

class DeltaCoder(Filter):
    def __init__(self):
        self.prev = 0

    def __call__(self, x):
        answer = x - self.prev
        self.prev = x
        return answer

class ZigZagCoder(Filter):
    def __call__(self, x):
        return (abs(x) << 1) | (1 if x < 0 else 0)

class Quantizer(Filter):
    def __init__(self, size):
        self.size = '1.' + ('0' * size)

    def __call__(self, x):
        return float(decimal.Decimal(x).quantize(decimal.Decimal(self.size)))

class FixedPoint(Filter):
    def __init__(self, fixed_point):
        self.fixed_point = fixed_point

    def __call__(self, x):
        return int(float(x) * self.fixed_point)

class Debugger(Filter):
    def __init__(self, ostream):
        self.log_file = ostream

    def __call__(self, x):
        self.log_file.write('%d\n' % x)
        return x

def quantize(args):
    quantize = Quantizer(args.bits)
    fixed_point_q = FixedPoint(args.fixed_point)
    if args.fixed_point == -1:
        answer_fn = quantize
    else:
        answer_fn = fixed_point_q
    if args.delta_coding:
        answer_fn = DeltaCoder().curry(answer_fn)
    if args.zig_zag_coding:
        answer_fn = ZigZagCoder().curry(answer_fn)
    if args.debug:
        answer_fn = Debugger(args.debug).curry(answer_fn)
    return answer_fn

def main(args):
    oobj = json.load(args.ifile, parse_float=quantize(args))
    if args.remove_spaces:
        args.ofile.write(json.dumps(oobj).replace(' ', ''))
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
    parser.add_argument('-z', '--zig-zag-coding', action='store_true')
    parser.add_argument('--debug', type=argparse.FileType('w'))
    main(parser.parse_args())
