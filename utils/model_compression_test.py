import sys
import argparse
import numpy

import h5py

def main(args):
    one, two = h5py.File(args.one, 'r'), h5py.File(args.two, 'r')
    assert set(one.keys()) == set(two.keys())
    for key in one.keys():
        assert set(one[key].keys()) == set(two[key].keys())
        for gkey in one[key].keys():
            dimensions = 5
            print('Testing', key, gkey)
            if args.almost_equal:
                numpy.testing.assert_array_almost_equal(
                    one[key][gkey][:dimensions], two[key][gkey][:dimensions],
                    decimal=args.precision)
            else:
                numpy.testing.assert_array_equal(
                    one[key][gkey][:dimensions], two[key][gkey][:dimensions])
            print('OK')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Test if two h5 files are the same-ish. ')
    parser.add_argument('one', help='Input file')
    parser.add_argument('two', help='Input file')
    parser.add_argument('--almost-equal', action='store_true')
    parser.add_argument('--precision', default=3, type=int)
    main(parser.parse_args())
