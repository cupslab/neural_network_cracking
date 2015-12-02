#!/usr/bin/env python

import sys
import argparse
import csv

def main(args):
    writer = csv.writer(args.ofile, delimiter='\t',
                        quoting=csv.QUOTE_NONE, quotechar=None)
    for row in csv.DictReader(
            args.ifile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None):
        if row['POLICY'] == args.policy:
            pwd, num = row['PASSWORD'], row[args.column]
            try:
                writer.writerow([pwd, num])
            except csv.Error as e:
                sys.stderr.write(
                    'Error writing %s:%s, contains delimiter\n' % (pwd, num))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Convert larger guessing table format')
    parser.add_argument('-i', '--ifile', type=argparse.FileType('r'),
                        help='Input file. Default is stdin. ',
                        default=sys.stdin)
    parser.add_argument('-o', '--ofile', type=argparse.FileType('w'),
                        help='Output file. Default is stdout. ',
                        default=sys.stdout)
    parser.add_argument('column')
    parser.add_argument('policy')
    main(parser.parse_args())
