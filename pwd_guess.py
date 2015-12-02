#!/usr/bin/env python3

# William Melicher
from __future__ import print_function
from keras.models import Sequential, slice_X, model_from_json
from keras.layers.core import Activation, Dense, RepeatVector
from keras.layers import recurrent
from sklearn.utils import shuffle
import numpy as np

import sys
import argparse
import itertools
import string
import math
import gzip
import csv
import logging
import cProfile
import json
import random
import pickle

class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    """
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

    @staticmethod
    def fromConfig(config):
        return CharacterTable(config.char_bag, config.max_len)

class ModelSerializer(object):
    def __init__(self, archfile = None, weightfile = None):
        self.archfile = archfile
        self.weightfile = weightfile
        self.model_creator_from_json = model_from_json

    def save_model(self, model):
        globals()['most_recent_model'] = model
        if self.archfile is None or self.weightfile is None:
            logging.info(
                'Cannot save model because file arguments were not provided')
            return
        logging.info('Saving model architecture')
        with open(self.archfile, 'w') as arch:
            arch.write(model.to_json())
        logging.info('Saving model weights')
        model.save_weights(self.weightfile, overwrite = True)

    def load_model(self):
        logging.info('Loading model architecture')
        with open(self.archfile, 'r') as arch:
            model = self.model_creator_from_json(arch.read())
        logging.info('Loading model weights')
        model.load_weights(self.weightfile)
        return model

PASSWORD_END = '\n'

class ModelDefaults(object):
    """
    Configuration information for guessing and training. Can be read from a file
    in json format.

    Attributes:
    char_bag - alphabet of characters over which to guess
    model_type - type of model. Read keras documentation for more types.
    hidden_size - size of each layer. More means better accuracy
    layers - number of hidden layers. More means better accuracy
    max_len - maximum length of any password in training data. This can be
              larger than all passwords in the data and the network may output
              guesses that are this many characters long.
    min_len - minimum length of any password that will be guessed
    training_chunk - Smaller training chunk means less memory consumed. This is
                     especially important because often, the blocking thing is
                     using memory on the GPU which is small.
    generations - More generations means it takes longer but is more accurate
    chunk_print_interval - Interval over which to print info to the log
    lower_probability_threshold - This controls how many passwords to output
                                  during generation. Lower means more passwords.
    relevel_not_matching_passwords - If true, then passwords that do not match
                                     the filter policy will have their
                                     probability equal to zero.
    generation_checkpoint - Every few generations, save the model.
    training_accuracy_threshold - If the accuracy is not improving by this
                                  amount, then quit.
    """
    char_bag = (string.ascii_lowercase +
                string.ascii_uppercase +
                string.digits +
                '~!@#$%^&*(),.<>/?\'"{}[]\|-_=+;: `' +
                PASSWORD_END)
    model_type = "JZS1"
    hidden_size = 128
    layers = 1
    max_len = 40
    min_len = 4
    training_chunk = 128
    generations = 20
    chunk_print_interval = 1000
    lower_probability_threshold = 10**-5
    relevel_not_matching_passwords = True
    generation_checkpoint = True
    training_accuracy_threshold = 10**-10
    train_test_ratio = 10

    def __init__(self, adict = None, **kwargs):
        self.adict = adict if adict is not None else dict()
        for k in kwargs:
            self.adict[k] = kwargs[k]

    def __getattribute__(self, name):
        if name != 'adict' and name in self.adict:
            return self.adict[name]
        else:
            return super().__getattribute__(name)

    @staticmethod
    def fromFile(afile):
        if afile is None:
            return ModelDefaults()
        with open(afile, 'r') as f:
            try:
                return ModelDefaults(json.load(f))
            except ValueError as e:
                logging.error('Error loading config. %s', str(e))
                sys.exit(1)

    def as_dict(self):
        answer = dict(vars(ModelDefaults).copy())
        answer.update(self.adict)
        return dict([(key, value) for key, value in answer.items() if (
            key[0] != '_' and not hasattr(value, '__call__')
            and not type(value) == staticmethod)])

    def model_type_exec(self):
        if self.model_type == 'JZS1':
            return recurrent.JZS1
        if self.model_type == 'GRU':
            return recurrent.GRU
        else:
            logging.warning('Unknown model type %s', self.model_type)
            return None

class Trainer(object):
    def __init__(self, pwd_list, config = ModelDefaults()):
        self.config = config
        self.chunk = 0
        self.generation = 0
        self.pwd_whole_list = list(pwd_list)
        self.ctable = CharacterTable.fromConfig(self.config)

    def train_chunk_from_pwds(self, pwds):
        def all_prefixes(pwd):
            return [pwd[:i] for i in range(len(pwd))] + [pwd]
        def all_suffixes(pwd):
            return [pwd[i] for i in range(len(pwd))] + [PASSWORD_END]
        def pad_to_len(astring):
            return astring + (
                PASSWORD_END * (self.config.max_len - len(astring)))
        return (
            map(pad_to_len, itertools.chain.from_iterable(
                map(all_prefixes, pwds))),
            itertools.chain.from_iterable(map(all_suffixes, pwds)))

    def next_train_chunk(self):
        if self.chunk * self.config.training_chunk >= len(self.pwd_whole_list):
            return [], []
        pwd_list = self.pwd_whole_list[
            self.chunk * self.config.training_chunk:
            min((self.chunk + 1) * self.config.training_chunk,
                len(self.pwd_whole_list))]
        self.chunk += 1
        return self.train_chunk_from_pwds(pwd_list)

    def sample_training_set(self, x_list, y_list, num = 10):
        for _ in range(num):
            idx = random.randrange(0, len(x_list))
            row_x, row_y = x_list[np.array([idx])], y_list[np.array([idx])]
            preds = self.model.predict(row_x, verbose = 0)
            q = self.ctable.decode(row_x[0])
            correct = self.ctable.decode(row_y[0])
            guess = preds[0][0]
            logging.debug('Sampling training set: %s, %s, %s',
                          q.strip(PASSWORD_END),
                          correct.strip(PASSWORD_END),
                          guess)

    def next_train_set_as_np(self):
        x_strings, y_strings = self.next_train_chunk()
        x_str_list, y_str_list = list(x_strings), list(y_strings)
        x_vec = np.zeros((len(x_str_list), self.config.max_len,
                          len(self.config.char_bag)), dtype = np.bool)
        for i, xstr in enumerate(x_str_list):
            x_vec[i] = self.ctable.encode(xstr, maxlen = self.config.max_len)
        y_vec = np.zeros((len(y_str_list), 1, len(self.config.char_bag)),
                         dtype = np.bool)
        for i, ystr in enumerate(y_str_list):
            y_vec[i] = self.ctable.encode(ystr, maxlen = 1)
        return shuffle(x_vec, y_vec)

    def build_model(self):
        model = Sequential()
        model.add(self.config.model_type_exec()(
            len(self.config.char_bag), self.config.hidden_size))
        model.add(RepeatVector(1))
        for _ in range(self.config.layers):
            model.add(self.config.model_type_exec()(
                self.config.hidden_size, self.config.hidden_size,
                return_sequences = True))
        model.add(Dense(self.config.hidden_size, len(self.config.char_bag)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer = 'adam')
        self.model = model

    def train_model(self, serializer):
        prev_accuracy = 0
        for gen in range(self.config.generations):
            self.generation = gen + 1
            logging.info('Generation ' + str(gen + 1))
            accuracy = self.train_model_generation()
            logging.info('Generation accuracy: %s', accuracy)
            if accuracy > prev_accuracy:
                serializer.save_model(self.model)
            if ((accuracy - prev_accuracy) <
                self.config.training_accuracy_threshold):
                logging.info('Accuracy diff of %s is less than threshold.',
                             accuracy - prev_accuracy)
                break
            prev_accuracy = accuracy

    def test_set(self, x_all, y_all):
        split_at = len(x_all) - max(
            int(len(x_all) / self.config.train_test_ratio), 1)
        x_train, x_val = (slice_X(x_all, 0, split_at), slice_X(x_all, split_at))
        y_train, y_val = (y_all[:split_at], y_all[split_at:])
        return x_train, x_val, y_train, y_val

    def training_step(self, x_all, y_all):
        x_train, x_val, y_train, y_val = self.test_set(x_all, y_all)
        train_loss, train_accuracy = self.model.train_on_batch(
            x_train, y_train, accuracy = True)
        test_loss, test_accuracy = self.model.test_on_batch(
            x_val, y_val, accuracy = True)
        if self.chunk % self.config.chunk_print_interval == 0:
            self.sample_training_set(x_val, y_val)
        return train_loss, train_accuracy, test_loss, test_accuracy

    def train_model_generation(self):
        self.chunk = 0
        random.shuffle(self.pwd_whole_list)
        total_chunks = math.ceil(
            len(self.pwd_whole_list) / self.config.training_chunk)
        logging.info('Total chunks %s', total_chunks)
        accuracy_accum = []
        x_all, y_all = self.next_train_set_as_np()
        while len(x_all) != 0:
            assert len(x_all) == len(y_all)
            tr_loss, _, te_loss, te_acc = self.training_step(x_all, y_all)
            accuracy_accum += [(len(x_all), te_acc)]
            if self.chunk % self.config.chunk_print_interval == 0:
                logging.info('Chunk %s of %s. Each chunk is size %s',
                             self.chunk, total_chunks, len(x_all))
                logging.info('Train loss %s. Test loss %s. Test accuracy %s.',
                             tr_loss, te_loss, te_acc)
            x_all, y_all = self.next_train_set_as_np()
        return sum(map(lambda x: x[0] * x[1], accuracy_accum)) / sum(
            map(lambda x: x[0], accuracy_accum))

    def train(self, serializer):
        logging.info('Building model...')
        self.build_model()
        logging.info('Done compiling model. Beginning training...')
        self.train_model(serializer)

class PwdList(object):
    def __init__(self, ifile_name):
        self.ifile_name = ifile_name

    def open_file(self):
        if self.ifile_name[-3:] == '.gz':
            return gzip.open(self.ifile_name, 'rt')
        return open(self.ifile_name, 'r')

    def as_list_iter(self, agen):
        return list([row.strip(PASSWORD_END) for row in agen])

    def as_list(self):
        answer = []
        ifile = self.open_file()
        try:
            answer = self.as_list_iter(ifile)
        finally:
            ifile.close()
        return answer

class TsvList(PwdList):
    def as_list_iter(self, agen):
        answer = []
        for row in csv.reader(agen, delimiter = '\t', quotechar = None):
            pwd, freq, _ = row
            for i in range(int(float.fromhex(freq))):
                answer.append(sys.intern(pwd))
        return answer

class Filterer(object):
    def __init__(self, config):
        self.char_bag = config.char_bag
        self.max_len = config.max_len
        self.min_len = config.min_len
        self.filtered_out = 0
        self.total = 0

    def signal_error(self, pwd):
        logging.warning('Not a valid password. %s\n' % pwd)

    def pwd_is_valid(self, pwd):
        pwd = pwd.strip(PASSWORD_END)
        answer = (all(map(lambda c: c in self.char_bag, pwd)) and
                  len(pwd) <= self.max_len and len(pwd) >= self.min_len)
        if not answer:
            self.filtered_out += 1
        self.total += 1
        return answer

    def filter_test(self, alist):
        for pwd in alist:
            if self.pwd_is_valid(pwd):
                self.signal_error(pwd)

    def filter(self, alist):
        return filter(self.pwd_is_valid, alist)

class Guesser(object):
    def __init__(self, model, config, ostream):
        self.model = model
        self.config = config
        self.ostream = ostream
        self.generated = 0
        self.ctable = CharacterTable.fromConfig(self.config)
        self.filterer = Filterer(self.config)

    def cond_prob_from_preds(self, char, preds):
        return preds[self.ctable.char_indices[char]]

    def relevel_prediction(self, preds, astring):
        if not self.filterer.pwd_is_valid(astring):
            preds[self.ctable.char_indices[PASSWORD_END]] = 0
        elif len(astring) == self.config.max_len:
            for c in self.config.char_bag:
                preds[self.ctable.char_indices[c]] = (
                    1 if c == PASSWORD_END else 0)
        sum_per = sum(preds)
        for i, v in enumerate(preds):
            preds[i] = v / sum_per

    def output_guess(self, password, prob):
        self.ostream.write('%s\t%s\n' % (
            password.strip(PASSWORD_END), prob))
        self.generated += 1

    def conditional_probs(self, astring):
        np_inp = np.zeros((1, self.config.max_len, len(self.config.char_bag)),
                          dtype = np.bool)
        np_inp[0] = self.ctable.encode(
            astring + (PASSWORD_END * (self.config.max_len - len(astring))))
        answer = self.model.predict(np_inp, verbose = 0)[0][0].copy()
        if self.config.relevel_not_matching_passwords:
            self.relevel_prediction(answer, astring.strip(PASSWORD_END))
        return answer

    def recur(self, astring, prob):
        if prob < self.config.lower_probability_threshold:
            return
        if len(astring) > self.config.max_len:
            if len(astring.strip(PASSWORD_END)) == self.config.max_len:
                self.output_guess(astring, prob)
            return
        prediction = self.conditional_probs(astring)
        for char in self.config.char_bag:
            chain_pass = astring + char
            chain_prob =  self.cond_prob_from_preds(char, prediction) * prob
            if (char == PASSWORD_END and
                chain_prob >= self.config.lower_probability_threshold):
                self.output_guess(chain_pass, chain_prob)
            elif char != PASSWORD_END:
                self.recur(chain_pass, chain_prob)

    def guess(self):
        self.recur('', 1)

    @staticmethod
    def do_guessing(model, config, ofname):
        ostream = open(ofname, 'w') if ofname != '-' else sys.stdout
        logging.info('Enumerating guesses...')
        guesser = Guesser(model, config, ostream)
        guesser.guess()
        ostream.close()
        logging.info('Generated %s guesses', guesser.generated)

log_level_map = {
    'info' : logging.INFO,
    'warning'  : logging.WARNING,
    'debug' : logging.DEBUG,
    'error' : logging.ERROR
}

def init_logging(args, config):
    log_format = '%(asctime)-15s %(levelname)s: %(message)s'
    log_level = log_level_map[args['log_level']]
    if args['log_file']:
        logging.basicConfig(filename = args['log_file'],
                            level = log_level, format = log_format)
    else:
        logging.basicConfig(level = log_level, format = log_format)
    logging.info('Beginning...')
    logging.info('Arguments %s', json.dumps(args, indent = 4))
    logging.info('Configuration %s', json.dumps(config.as_dict(), indent = 4))

def train(args, config):
    global trained_model
    if args['tsv']:
        input_const = TsvList
    else:
        input_const = PwdList
    ilist = input_const(args['pwd_file']).as_list()
    filt = Filterer(config)
    if args['test_input']:
        logging.info('filtering only...')
        filt.filter_test(ilist)
    else:
        logging.info('Starting training...')
        plist = list(filt.filter(ilist))
        logging.info('Done reading passwords...')
        if len(plist) == 0:
            logging.error('Empty training set! Quiting...')
            sys.exit(1)
        logging.info('Saving model')
        trainer = Trainer(plist, config)
        trainer.train(ModelSerializer(args['arch_file'], args['weight_file']))
        if args['enumerate_ofile']:
            Guesser.do_guessing(trainer.model, config, args['enumerate_ofile'])
    logging.info('Filtered %s of %s passwords', filt.filtered_out, filt.total)

def guess(args, config):
    logging.info('Loading model...')
    if args['arch_file'] is None or args['weight_file'] is None:
        logging.error('Architecture file or weight file not found. Quiting...')
        sys.exit(1)
    Guesser.do_guessing(ModelSerializer(
        args['arch_file'], args['weight_file']).load_model(),
        config, args['enumerate_ofile'])

def main(args):
    if args['help_config']:
        sys.stdout.write(ModelDefaults.__doc__ + '\n')
        sys.exit(0)
    config = ModelDefaults.fromFile(args['config'])
    init_logging(args, config)
    if args['pwd_file']:
        train(args, config)
    elif args['enumerate_ofile']:
        guess(args, config)
    if not args['pwd_file'] and not args['enumerate_ofile']:
        logging.error('Nothing to do! Use --pwd-file or --enumerate-ofile. ')
        sys.exit(1)
    logging.info('Done!')

def make_parser():
    parser = argparse.ArgumentParser(
        description=('Neural Network with passwords. This program uses a '
                     'neural network to guess passwords. This happens in two'
                     ' phases, training and enumeration. Either --pwd-file or'
                     ' --enumerate-ofile are required. --pwd-file will give a'
                     ' password file as training data. --enumerate-ofile will'
                     ' guess passwords based on an existing model. '))
    parser.add_argument('--pwd-file',
                        help=('Input file name. Will be interpreted as a '
                              'gziped file if this argument ends in `.gz\'. '))
    parser.add_argument('--arch-file',
                        help = 'Output file for the model architecture. ')
    parser.add_argument('--weight-file',
                        help = 'Output file for the weights of the model. ')
    parser.add_argument('--tsv', action='store_true',
                        help=('Input file from --pwd-file is in TSV format. '
                              'The first column of the TSV should be the'
                              ' password. '))
    parser.add_argument('--test-input', action='store_true', help=(
        'Test if the password input is valid and print to stderr errors. Will'
        ' not train the neural network. '))
    parser.add_argument('--enumerate-ofile',
                        help = 'Enumerate guesses output file')
    parser.add_argument('--config', help = 'Config file in json. ')
    parser.add_argument('--profile',
                        help = 'Profile execution and save to the given file. ')
    parser.add_argument('--help-config', action = 'store_true',
                        help = 'Print help for config files and exit')
    parser.add_argument('--log-file')
    parser.add_argument('--log-level', default = 'info',
                        choices = ['debug', 'info', 'warning', 'error'])
    return parser

if __name__=='__main__':
    args = vars(make_parser().parse_args())
    main_bundle = lambda: main(args)
    if args['profile'] is not None:
        cProfile.run('main_bundle()', filename = args['profile'])
    else:
        main_bundle()
