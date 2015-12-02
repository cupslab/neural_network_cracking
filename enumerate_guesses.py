#!/usr/bin/env python

import sys
import argparse

import pwd_guess

def main(args):
    model = pwd_guess.ModelSerializer(
        args['architecture'], args['modelweights']).load_model()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Enumerate password guesses from a trained NN')
    parser.add_argument('architecture',
                        help = ('Input file for the network architecture. '
                                'Should be a json file. '))
    parser.add_argument('modelweights',
                        help = ('Input file for the model weights. Should be an'
                                ' h5 file. '))
    parser.add_argument('ofile', help = 'Output file. ')
    main(vars(parser.parse_args()))
