import sys
import argparse
import logging

import pwd_guess

def main(args):
    pwd_guess.init_logging(args)
    logging.info('Loading model...')
    model = pwd_guess.ModelSerializer(
        args['architecture'], args['modelweights']).load_model()
    config = pwd_guess.ModelDefaults.fromFile(args['config'])
    ostream = open(args['ofile'], 'w')
    logging.info('Enumerating guesses...')
    guesser = pwd_guess.Guesser(model, config, ostream)
    try:
        guesser.guess()
    finally:
        ostream.close()

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
    parser.add_argument('--config', help = 'Configuration file in json. ')
    parser.add_argument('--log-file')
    main(vars(parser.parse_args()))
