import sys
import argparse
import json

def main(args):
    json.dump([line.strip('\n') for line in args.ifile], args.ofile)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--ifile',
                        type = argparse.FileType('r'),
                        help = 'Input file. Default is stdin. ',
                        default = sys.stdin)
    parser.add_argument('-o', '--ofile',
                        type = argparse.FileType('w'),
                        help = 'Output file. Default is stdout. ',
                        default = sys.stdout)
    main(parser.parse_args())
