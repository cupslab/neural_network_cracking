import sys
import argparse
import pstats

def main(args):
    stat = pstats.Stats(args.ifile)
    stat.sort_stats('tottime')
    stat.print_stats()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Read output of cProfile. ')
    parser.add_argument('ifile', help = 'Input file. ')
    main(parser.parse_args())
