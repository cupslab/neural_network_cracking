import sys
import argparse
import csv
import os
import collections

def guess_numbers(ifname):
    with open(ifname, 'r') as ifile:
        try:
            return dict([(row[2], int(row[5])) for row in csv.reader(
                ifile, delimiter='\t', quotechar=None)])
        except ValueError as e:
            sys.stderr.write('ERROR!!! %s %s\n' % (ifname, str(e)))
            sys.exit(1)

def main(args):
    gns = guess_numbers(args.files[0])
    for f in args.files[1:]:
        for pwd, value in guess_numbers(f).items():
            gns[pwd] = min(
                gns[pwd] if pwd in gns and gns[pwd] >= 0 else float('inf'),
                value if value >= 0 else float('inf'))
    with open(args.pwd_file, 'r') as pwds:
        counts = collections.Counter([line.strip(os.linesep) for line in pwds])
    with open(os.path.join(args.odir, 'totalcounts.' + args.name), 'w') as tot:
        tot.write('%s:Total count\t%d\n' % (args.name, max(
            filter(lambda x: x != float('inf'), gns.values()))))
    with open(os.path.join(args.odir, 'lookupresults.' + args.name), 'w') as lr:
        writer = csv.writer(lr, delimiter='\t', quotechar=None)
        for pwd, value in gns.items():
            for _ in range(counts[pwd]):
                writer.writerow([
                    'no_user', args.name, pwd, '0x0.1p-1', '', value, ''])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('pwd_file')
    parser.add_argument('files', nargs='+')
    parser.add_argument('--odir', default='./')
    main(parser.parse_args())
