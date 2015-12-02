#!/usr/bin/env python
# Uses reservoir sampling algorithm:
# https://en.wikipedia.org/wiki/Reservoir_sampling and
# http://data-analytics-tools.blogspot.com/2009/09/reservoir-sampling-algorithm-in-perl.html

import sys
import argparse
import random
import collections

import pwd_guess as pg

TEST_SUFFIX = '.test'
TRAIN_SUFFIX = '.train'

def pwds(ifile, fmt, config_file):
    config = pg.ModelDefaults.fromFile(config_file)
    pwd_lister = pg.PwdList.getFactory(fmt, config)(ifile)
    filterer = pg.Filterer(config)
    return filterer.filter(pwd_lister.as_list())

def main(args):
    sample_buffer = []
    for i, pwd in enumerate(pwds(args.ifile, args.format, args.config)):
        pwd, weight = pwd
        for j in range(weight):
            num_seen = i + j
            if num_seen < args.num:
                sample_buffer.append(pwd)
            elif (num_seen >= args.num and
                  random.random() < args.num / float(num_seen + 1)):
                sample_buffer[random.randint(0, len(sample_buffer) - 1)] = pwd
    ofname = args.oprefix
    with open(ofname + TEST_SUFFIX, 'w') as ofile:
        for pwd in sample_buffer:
            ofile.write('%s\n' % pwd)
    sample_buf_pwds = collections.Counter(sample_buffer)
    with open(ofname + TRAIN_SUFFIX, 'w') as ofile:
        for pwd_tup in pwds(args.ifile, args.format, args.config):
            pwd, weight = pwd_tup
            new_weight = weight - sample_buf_pwds[weight]
            ofile.write('%s\t%s\n' % (pwd, float.hex(float(new_weight))))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate test data')
    parser.add_argument('-i', '--ifile', help='Input file (s). ',
                        nargs='+', required=True)
    parser.add_argument('-f', '--format', help='Format of input file (s). ',
                        nargs='+', required=True)
    parser.add_argument('-o', '--oprefix', required = True,
                        help=('Output file prefix. '))
    parser.add_argument('-c', '--config', help='Config file. ', required=True)
    parser.add_argument('-n', '--num', type=int, default=10,
                        help='Number of passwords to sample. ')
    main(parser.parse_args())
