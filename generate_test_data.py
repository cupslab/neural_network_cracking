import sys
import argparse
import random

distributions = {
    'constant' : lambda: 'aaa'
}

def main(args):
    for _ in range(args.line_count):
        args.ofile.write('%s\n' % distributions[args.distribution]())
    args.ofile.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Generate test data of a's and b's")
    parser.add_argument('-o', '--ofile',
                        type = argparse.FileType('w'),
                        help = 'Input file. Default is stdout. ',
                        default = sys.stdout)
    parser.add_argument('--line-count', type=int, default = 100)
    parser.add_argument('distribution', choices = distributions.keys())
    main(parser.parse_args())
