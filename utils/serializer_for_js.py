import sys
import argparse

import serialize_keras

def main(args):
    serialize_keras.serialize(args.model_file, args.weight_file,
                              args.ofile, args.compress)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description=('Serialize a model for JS. Must have the '
                     'serialize_keras.py script in the same'
                     ' directory or in PYTHONPATH. You can retrieve it from '
                     'https://github.com/scienceai/neocortex. '))
    parser.add_argument('model_file')
    parser.add_argument('weight_file')
    parser.add_argument('ofile')
    parser.add_argument('-c', '--compress', action='store_true')
    main(parser.parse_args())
