#!/usr/bin/env python
# Uses reservoir sampling algorithm:
# https://en.wikipedia.org/wiki/Reservoir_sampling and
# http://data-analytics-tools.blogspot.com/2009/09/reservoir-sampling-algorithm-in-perl.html

import sys
import argparse
import random

import pwd_guess as pg

def main(args):
    config = pg.ModelDefaults.fromFile(args.config)
    pwd_lister = pg.PwdList.getFactory(args.format, config)(args.ifile)
    filterer = pg.Filterer(config)
    sample_buffer = []
    for i, pwd in enumerate(filterer.filter(pwd_lister.as_list())):
        pwd, weight = pwd
        for j in range(weight):
            num_seen = i + j
            if num_seen < args.num:
                sample_buffer.append(pwd)
            elif (num_seen >= args.num and
                  random.random() < args.num / float(num_seen + 1)):
                sample_buffer[random.randint(0, len(sample_buffer) - 1)] = pwd
    ofname = config.password_test_fname
    if args.ofile:
        ofname = args.ofile
    with open(ofname, 'w') as ofile:
        for pwd in sample_buffer:
            ofile.write('%s\n' % pwd)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate test data')
    parser.add_argument('-i', '--ifile', help='Input file (s). ',
                        nargs='+', required=True)
    parser.add_argument('-f', '--format', help='Format of input file (s). ',
                        nargs='+', required=True)
    parser.add_argument('-o', '--ofile',
                        help=('Output file. If not given, will use '
                              '"password_test_fname" from the config file. '))
    parser.add_argument('-c', '--config', help='Config file. ', required=True)
    parser.add_argument('-n', '--num', type=int, default=10,
                        help='Number of passwords to sample. ')
    main(parser.parse_args())
