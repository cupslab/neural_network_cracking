# author: William Melicher
import argparse
import collections
import json
import sys

import numpy as np
import pwd_guess as pg
import logging

PASSWORD_START = '\t';

DEFAULT_CONFIG = {
    'additive_smoothing_amount' : 1,
    'backoff_smoothing_threshold' : 10
}

class NoSmoothingSmoother(object):
    def __init__(self, freq_dict, config):
        self.alphabet = sorted(config.char_bag)
        self.freq_dict = freq_dict
        self.config = config

    def predict(self, ctx_arg, answer):
        assert answer.shape == (len(self.alphabet), )
        # TODO: memoize computation here
        return self._predict(ctx_arg, answer)

    def _predict(self, ctx_arg, answer):
        total_sum = 0
        for i, next_char in enumerate(self.alphabet):
            ngram = ctx_arg + next_char
            freq = self.freq_dict[ngram] if ngram in self.freq_dict else 0
            answer[i] += freq
            total_sum += freq
        for i in range(len(self.alphabet)):
            answer[i] /= total_sum

class AdditiveSmoothingSmoother(NoSmoothingSmoother):
    def __init__(self, freq_dict, config):
        super().__init__(freq_dict, config)
        self.amount = self.config.additive_smoothing_amount

    def _predict(self, ctx_arg, answer):
        total_sum = 0
        for i, next_char in enumerate(self.alphabet):
            ngram = ctx_arg + next_char
            freq = (self.freq_dict[ngram] + self.amount
                    if ngram in self.freq_dict else self.amount)
            answer[i] += freq
            total_sum += freq
        for i in range(len(self.alphabet)):
            answer[i] /= total_sum

class BackoffSmoother(NoSmoothingSmoother):
    def __init__(self, freq_dict, config):
        super().__init__(freq_dict, config)
        self.threshold = self.config.backoff_smoothing_threshold

    def _predict(self, ctx_arg, answer):
        total_sum = 0
        for i, next_char in enumerate(self.alphabet):
            ngram = ctx_arg + next_char
            freq = self.freq_dict[ngram] if ngram in self.freq_dict else 0
            if freq < self.threshold:
                freq = 0
            answer[i] += freq
            total_sum += freq
        if total_sum == 0:
            answer.fill(0)
            assert len(ctx_arg) != 0, 'Backing off on 0 character string!?!'
            self.predict(ctx_arg[1:], answer)
            return
        for i in range(len(self.alphabet)):
            answer[i] /= total_sum

class MarkovModel(object):
    LOGGING_FREQUENCY = 1000000

    SMOOTHING_MAP = {
        'none' : NoSmoothingSmoother,
        'additive' : AdditiveSmoothingSmoother,
        'backoff' : BackoffSmoother
    }

    def __init__(self, config, smoothing='none', order=2):
        self.alphabet = sorted(config.char_bag)
        self.chars_to_index = dict([
            (c, i) for i, c in enumerate(self.alphabet)])
        self.smoothing = smoothing
        self.freq_dict = collections.defaultdict(int)
        self.order = order
        self.config = config
        self.smoother = None
        assert pg.PASSWORD_END in self.alphabet

    def make_smoother(self):
        return self.SMOOTHING_MAP[self.smoothing](self.freq_dict, self.config)

    def train_on_pwd(self, pwd, freq):
        pwd_len_plus_one = len(pwd) + 1
        for j in range(1, min(self.order, pwd_len_plus_one)):
            self.increment(pwd[:j], freq)
        for i in range(pwd_len_plus_one - self.order):
            self.increment(pwd[i:i + self.order], freq)
        self.increment(pwd[-self.order + 1:] + pg.PASSWORD_END, freq)

    def train(self, pwds):
        ctr = 0
        for pwd, freq in pwds:
            ctr += 1
            if ctr % self.LOGGING_FREQUENCY == 0:
                logging.info('Training on password %d', ctr)
            self.train_on_pwd(pwd, freq)
        self.smoother = self.make_smoother()

    def increment(self, pwd, freq):
        assert freq != 0
        assert len(pwd) <= self.order
        self.freq_dict[pwd] += freq

    def truncate_context(self, context):
        if len(context) >= self.order:
            return context[-(self.order - 1):]
        return context

    def probability_next_char(self, context, nc):
        assert nc in self.chars_to_index, (
            '%s not in alphabet. Please change config file' % nc)
        probs = np.zeros((len(self.alphabet), ), dtype=np.float64)
        self.predict(context, probs)
        return probs[self.chars_to_index[nc]]

    def predict(self, context, answer):
        return self.smoother.predict(self.truncate_context(context), answer)

    def saveModel(self, fname):
        logging.info('Saving model to %s', fname)
        with open(fname, 'w') as ofile:
            json.dump(self.freq_dict, ofile)

    @classmethod
    def fromModelFile(cls, fname, config, smoothing='none', order=2):
        logging.info('Loading model from %s', fname)
        with open(fname, 'r') as ifile:
            oobj = json.load(ifile)
        answer = cls(config, smoothing=smoothing, order=order)
        answer.freq_dict = oobj
        answer.smoother = answer.make_smoother()
        return answer

class BackoffMarkovModel(MarkovModel):
    def __init__(self, config, smoothing='backoff', order=2):
        super().__init__(config, smoothing, order)
        assert smoothing == 'backoff', ('Backoff Markov Model must be created '
                                        'with backoff smoothing')
        self.alphabet += PASSWORD_START

    def train_on_pwd(self, pwd, freq):
        pwd_norm = PASSWORD_START + pwd + pg.PASSWORD_END
        pwd_len = len(pwd_norm)
        for pwd_idx in range(pwd_len):
            pwd_idx_plus_one = pwd_idx + 1
            for order_idx in range(min(self.order, pwd_len - pwd_idx)):
                self.increment(pwd_norm[
                    pwd_idx:pwd_idx_plus_one + order_idx], freq)

class MarkovModelBuilder(object):
    def __init__(self, config,
                 smoothing = 'none', order = 2, model_file = None):
        self.config = config
        self.smoothing = smoothing
        self.order = order
        self.model_file = None

    def build(self):
        cls = MarkovModel
        if self.smoothing == 'backoff':
            cls = BackoffMarkovModel
        if self.model_file is not None:
            return cls.fromModelFile(self.model_file, self.config,
                                     smoothing=self.smoothing, order=self.order)
        else:
            return cls(self.config, smoothing=self.smoothing, order=self.order)

class MarkovGuessingFunction(object):
    def conditional_probs_many(self, astring_list):
        answer = np.zeros((len(astring_list), 1, self.ctable.vocab_size),
                          dtype=np.float64)
        for i, astring in enumerate(astring_list):
            self.model.predict(astring, answer[i, 0])
        if self.relevel_not_matching_passwords:
            self.relevel_prediction_many(answer, astring_list)
        return answer

class MarkovGuesser(MarkovGuessingFunction, pg.Guesser):
    pass

class MarkovRandomWalkGuesser(MarkovGuessingFunction, pg.RandomWalkGuesser):
    pass

class MarkovRandomWalkDelAmico(MarkovGuessingFunction, pg.RandomWalkDelAmico):
    pass

class MarkovRandomGenerator(MarkovGuessingFunction, pg.RandomGenerator):
    pass

MARKOV_GUESSER_MAP = {
    'markov_random_walk' : MarkovRandomWalkGuesser,
    'markov_delamico_random_walk' : MarkovRandomWalkDelAmico,
    'markov_human' : MarkovGuesser,
    'markov_generate_random' : MarkovRandomGenerator,
}

def read_config(fname):
    if fname is not None:
        logging.info('Reading config from %s', fname)
        answer = pg.ModelDefaults.fromFile(fname)
    else:
        logging.info('Using default config')
        # Default should be to use simulated frequency optimization
        answer = pg.ModelDefaults(guesser_class='markov_human',
                                  simulated_frequency_optimization=True)
    for key in DEFAULT_CONFIG:
        if key not in answer.adict:
            answer.adict[key] = DEFAULT_CONFIG[key]
    answer.validate()
    logging.info('Using config: %s', json.dumps(answer.as_dict(), indent=4))
    return answer

def train(args):
    logging.info('Beginning training of %s-gram model...', args.k_order)
    config = read_config(args.config)
    model = MarkovModelBuilder(
        config, order = args.k_order, smoothing = args.smoothing).build()
    model.train(pg.ResetablePwdList(
        [args.train_file], [args.train_format], config).as_iterator(quick=True))
    model.saveModel(args.ofile)

def make_guesser_builder(args):
    config = read_config(args.config)
    if config.guesser_class not in MARKOV_GUESSER_MAP:
        logging.critical(('Configuration option guesser_class is %s must be '
                          'one of: %s'), config.guesser_class,
                         ", ".join(sorted(list(MARKOV_GUESSER_MAP.keys()))))
        sys.exit(1)
    if args.model_file is None:
        logging.critical('Must provide --model-file argument! Exiting...')
        sys.exit(1)
    if args.ofile is None:
        logging.critical('Must provide ofile argument! Exiting...')
        sys.exit(1)

    config.password_test_fname = args.password_file
    guesser_builder = pg.GuesserBuilder(config)
    guesser_builder.add_model(MarkovModelBuilder(
        config, smoothing=args.smoothing, order=args.k_order,
        model_file=args.model_file).build())
    guesser_builder.add_file(args.ofile)
    guesser_builder.other_class_builders = MARKOV_GUESSER_MAP
    return guesser_builder.build()

def main(args):
    pg.init_logging(vars(args))
    if args.train_file is not None:
        train(args)
    elif args.model_file is not None:
        guesser = make_guesser_builder(args)
        if args.password_file is None:
            guesser.complete_guessing()
        else:
            guesser.calculate_probs()
    else:
        logging.error('Must provide --train-file or --model-file flag. ')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Train and guess with a markov model. ')
    parser.add_argument('-t', '--train-file',
                        help='Training file. Will train a model. ')
    parser.add_argument('-o', '--ofile', help='Output file. ')
    parser.add_argument('-m', '--model-file',
                        help='Model file. Will guess passwords. ')
    parser.add_argument('-p', '--password-file',
                        help='Password file. Will calculate probabilities. ')
    parser.add_argument('-k', '--k-order', type=int, default=2,
                        help=('Giving an argument of 2 means using 1 '
                              'character of context to predict the next '
                              'character. Default is 2. '))
    parser.add_argument('-c', '--config', help='Config file. ')
    parser.add_argument('-s', '--smoothing', default = 'none',
                        help='Type of smoothing. Default is no smoothing. ',
                        choices=sorted(MarkovModel.SMOOTHING_MAP.keys()))
    parser.add_argument('-f', '--train-format',
                        help='Can be list or tsv. Default is tsv',
                        choices=['list', 'tsv'], default='tsv')
    parser.add_argument('-l', '--log-file')
    parser.add_argument('--log-level', default='info', choices=pg.log_level_map)
    main(parser.parse_args())
