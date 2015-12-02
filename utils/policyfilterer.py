#!/usr/bin/env python

import sys
import argparse
import os

import pwd_guess

def main(args):
    policy = pwd_guess.policy_list[args.policy]
    fmt = '%%s%s' % os.linesep
    totalcount, filteredcount = 0, 0
    with open(args.ifile, 'r', encoding=args.encoding) as ifile:
        for line in ifile:
            totalcount += 1
            pwd = line.strip(os.linesep)
            if policy.pwd_complies(pwd):
                filteredcount += 1
                args.ofile.write(fmt % pwd)
    args.ofile.close()
    sys.stderr.write('Total passwords: %d, Outputed passwords: %d\n' % (
        totalcount, filteredcount))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--ofile', type = argparse.FileType('w'),
                        help = 'Output file. Default is stdout. ',
                        default = sys.stdout)
    parser.add_argument('-e', '--encoding', default='utf8',
                        help='Encoding. Default is utf8. ')
    parser.add_argument('policy', choices=sorted(pwd_guess.policy_list.keys()))
    parser.add_argument('ifile', help = 'Input file. ')
    main(parser.parse_args())
