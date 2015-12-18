import sys
import argparse

def main(args):
    pwds = set([line.strip('\n') for line in args.pfile])
    seen = set()
    ctr = 0
    for ctr, line in enumerate(args.ifile):
        pwd, prob = line.strip('\n').split('\t')
        ctr += 1
        if pwd in seen:
            ctr -= 1
            continue
        if pwd in pwds:
            seen.add(pwd)
            args.ofile.write('%s\t%s\t%s\n' % (pwd, ctr + 1, prob))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--ifile',
                        type = argparse.FileType('r'),
                        help = 'Input file. Default is stdin. ',
                        default = sys.stdin)
    parser.add_argument('pfile',
                        type = argparse.FileType('r'),
                        help = 'Input test file. ')
    parser.add_argument('-o', '--ofile',
                        type = argparse.FileType('w'),
                        help = 'Output file. Default is stdout. ',
                        default = sys.stdout)
    main(parser.parse_args())
