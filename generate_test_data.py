import sys
import argparse
import random

import numpy as np

skewed_dict = [
    (1, 'abab'),
    (4, 'abbbb'),
    (2, 'aaaa'),
    (3, 'aaab')
]

sums = sum(map(lambda x: x[0], skewed_dict))
probs = list(map(lambda x: x[0] / sums, skewed_dict))

def skewed():
    return skewed_dict[np.random.choice(len(skewed_dict), 1, p = probs)[0]][1]

def subtract():
    a, b = random.randint(0, 10), random.randint(0, 10)
    return str(a) + str(b) + str(abs(a - b))

distributions = {
    'constant' : lambda: 'aaa',
    'skewed' : skewed,
    'subtract' : subtract
}

def main(args):
    for _ in range(args.line_count):
        args.ofile.write('%s\n' % distributions[args.distribution]())
    args.ofile.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Generate test data")
    parser.add_argument('-o', '--ofile',
                        type = argparse.FileType('w'),
                        help = 'Input file. Default is stdout. ',
                        default = sys.stdout)
    parser.add_argument('--line-count', type=int, default = 100)
    parser.add_argument('distribution', choices = distributions.keys())
    main(parser.parse_args())
