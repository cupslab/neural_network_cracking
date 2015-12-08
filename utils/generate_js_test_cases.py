#!/usr/bin/env python

import sys
import argparse
import json
import logging

import pwd_guess

CONDITIONAL_PROB_TESTS = [
    'password12',
    'P@$$W0rd12',
    '$$QWERT',
    '111111',
    '1111111111111111111',
    '11111111111111',
    'asdfasdfasd',
    'viua4n 84y',
    'Viua4n 84y'
]

def main(args):
    with open(args.config, 'r') as config_args:
        config = pwd_guess.ModelDefaults(json.load(config_args)['config'])
    guesser = pwd_guess.Guesser(pwd_guess.ModelSerializer(
        args.model_file, args.weight_file).load_model(), config, None)
    prob_calculator_prefix = pwd_guess.ProbabilityCalculator(
        guesser, prefixes = True)
    assert prob_calculator_prefix.template_probs
    prob_calculator_noprefix = pwd_guess.ProbabilityCalculator(guesser)
    assert prob_calculator_noprefix.template_probs
    cond_prob_tests_tuples = list(map(lambda x: (x, 1), CONDITIONAL_PROB_TESTS))
    json.dump({
        'test_data': cond_prob_tests_tuples,
        'test_case1_conditional_prob': list(map(
            lambda x: x.tolist(), map(guesser.conditional_probs,
                                      CONDITIONAL_PROB_TESTS))),
        'test_case2_total_prob_template_prefix': list(
            prob_calculator_prefix.calc_probabilities(cond_prob_tests_tuples)),
        'test_case3_total_prob_template_noprefix': (
            list(prob_calculator_noprefix.calc_probabilities(
                cond_prob_tests_tuples)))
    }, args.output)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Generated test cases for a the JS model.')
    parser.add_argument('model_file')
    parser.add_argument('weight_file')
    parser.add_argument('config')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        help='Output json file default is stdout. ',
                        default=sys.stdout)
    main(parser.parse_args())
