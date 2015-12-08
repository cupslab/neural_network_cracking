import sys
import argparse
import json
import csv

def main(args):
    output = []
    with open(args.ifile, 'r') as ifile:
        for row in csv.reader(ifile, quotechar=None, delimiter='\t'):
            output.append((float(row[1]), float(row[2])))
    json.dump({
        'guessing_table' : sorted(output, key=lambda x: x[0])[::10]
    }, args.ofile)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Extract guess numbers from probabilities')
    parser.add_argument('ifile')
    parser.add_argument('--ofile', type=argparse.FileType('w'),
                        default=sys.stdout)
    main(parser.parse_args())
