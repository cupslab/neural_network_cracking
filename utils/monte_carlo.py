import sys
import argparse
import csv

try:
    import pwd_guess
except ImportError as e:
    sys.stderr.write(('Error importing pwd_guess: %s\n'
                      'Make sure that pwd_guess is in your PYTHONPATH\n' %
                      str(e)))
    sys.exit(1)

def main(args):
    input_probs = []
    prob_fmt = float.fromhex if args.hex else float
    for pwd, prob in csv.reader(args.testfile, delimiter='\t', quotechar=None):
        prob_v = prob_fmt(prob)
        if prob_v < 1 and prob_v >= 0:
            input_probs.append( (pwd, prob_v) )
    calculator = pwd_guess.DelAmicoCalculator(
        args.ofile, input_probs,
        pwd_guess.ModelDefaults(
            random_walk_confidence_bound_z_value=args.confidence_interval))
    for row in csv.reader(args.randomfile, delimiter='\t', quotechar=None):
        prob_str, pwd = row
        prob = prob_fmt(prob_str)
        if prob >= 0:
            calculator.serialize(pwd, prob)
        if prob >= 1:
            calculator.serialize(pwd, 0)
    calculator.finish()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description=('Takes randomly generated passwords as input, and test'
                     ' password probabilities and calculates guess numbers. '))
    parser.add_argument('randomfile', type=argparse.FileType('r'),
                        help=('Randomly generated passwords file. Should be a '
                              'tsv where the first column is probability and '
                              'second column is the password. '))
    parser.add_argument('testfile', type=argparse.FileType('r'),
                        help=('Password file. Should be a tsv of passwords '
                              'where the first column is the probability and '
                              'second is the password. '))
    parser.add_argument('-o', '--ofile', type=argparse.FileType('w'),
                        help='Input file. Default is stdout. ',
                        default=sys.stdout)
    parser.add_argument('-c', '--confidence-interval', type=float, default=1.96,
                        help=('Float of the confidence bound. Should be a '
                              'lookup in the theta table. Default is 1.96 '
                              'which corresponds to a 95 percent confidence '
                              'interval. '))
    parser.add_argument('--hex', action='store_true',
                        help='Probabilities are in hex format. ')
    main(parser.parse_args())
