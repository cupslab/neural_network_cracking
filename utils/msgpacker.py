import sys
import argparse
import msgpack
import json

def main(args):
    msgpack.dump(json.load(args.ifile), args.ofile,
                 use_single_float=args.single_precision_float)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--ifile',
                        type = argparse.FileType('r'),
                        help = 'Input file. Default is stdin. ',
                        default = sys.stdin)
    parser.add_argument('-o', '--ofile',
                        type = argparse.FileType('wb'),
                        help = 'Output file. Default is stdout. ',
                        default = sys.stdout)
    parser.add_argument('-s', '--single-precision-float', action='store_true')
    main(parser.parse_args())
