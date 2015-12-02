from __future__ import print_function
from keras.models import Sequential, slice_X
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


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


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


class ModelSerializer(object):
    def __init__(self, archfile, weightfile):
        self.archfile = archfile
        self.weightfile = weightfile

    def save_model(self, model):
        with open(self.archfile, 'w') as arch:
            arch.write(model.to_json())
        model.save_weights(self.weightfile)


class ModelDefaults(object):
    default_char_bag = (string.ascii_lowercase +
                        string.ascii_uppercase +
                        string.digits +
                        '~!@#$%^&*(),.<>/?\'"{}[]\|-_=+;:\n `')
    HIDDEN_SIZE = 128
    LAYERS = 1
    TRAIN_TEST_SPLIT = 0.1
    BATCH_SIZE = 128
    MAX_LEN = 40
    MIN_LEN = 4
    TRAINING_CHUNK = 100000
    generations = 20


class Trainer(object):
    def __init__(self, pwd_list,
                 max_len = ModelDefaults.MAX_LEN,
                 char_bag = ModelDefaults.default_char_bag,
                 hidden_size = ModelDefaults.HIDDEN_SIZE,
                 layers = ModelDefaults.LAYERS,
                 train_test_split = ModelDefaults.TRAIN_TEST_SPLIT,
                 batch_size = ModelDefaults.BATCH_SIZE,
                 model_type = recurrent.JZS1,
                 generations = ModelDefaults.generations,
                 visualize_errors = True,
                 training_chunk = ModelDefaults.TRAINING_CHUNK):
        # Tuning parameters
        self.max_len = max_len
        self.char_bag = char_bag
        self.hidden_size = hidden_size
        self.layers = layers
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.model_type = model_type
        self.visualize_errors = visualize_errors
        self.generations = generations
        self.training_chunk = training_chunk

        # Variables
        self.pwd_whole_list = list(pwd_list)
        self.chunk = 0
        self.ctable = self.make_char_table()

    def train_chunk_from_pwds(self, pwds):
        def all_prefixes(pwd):
            return [pwd[:i] for i in range(len(pwd))] + [pwd]
        def all_suffixes(pwd):
            return [pwd[i] for i in range(len(pwd))] + ['\n']
        def pad_to_len(astring):
            return astring + ('\n' * (self.max_len - len(astring)))
        return (
            map(pad_to_len, itertools.chain.from_iterable(
                map(all_prefixes, pwds))),
            itertools.chain.from_iterable(map(all_suffixes, pwds)))

    def next_train_chunk(self):
        logging.info('Training chunk: %s', self.chunk)
        pwd_list = self.pwd_whole_list[
            self.chunk * self.training_chunk:
            (self.chunk + 1) * self.training_chunk]
        self.chunk += 1
        return self.train_chunk_from_pwds(pwd_list)

    def make_char_table(self):
        return CharacterTable(self.char_bag, self.max_len)

    def next_train_set_as_np(self):
        x_strings, y_strings = self.next_train_chunk()
        x_str_list, y_str_list = list(x_strings), list(y_strings)
        x_vec = np.zeros((len(x_str_list), self.max_len, len(self.char_bag)),
                         dtype = np.bool)
        for i, xstr in enumerate(x_str_list):
            x_vec[i] = self.ctable.encode(xstr, maxlen = self.max_len)
        y_vec = np.zeros((len(y_str_list), 1, len(self.char_bag)),
                         dtype = np.bool)
        for i, ystr in enumerate(y_str_list):
            y_vec[i] = self.ctable.encode(ystr, maxlen = 1)
        return (x_vec, y_vec)

    def verification_set(self):
        logging.info('Preprocessing training data')
        x_vec, y_vec = self.next_train_set_as_np()
        logging.info('Creating verification and training sets')
        X, y = shuffle(x_vec, y_vec)
        split_at = len(X) - math.ceil(len(X) * self.train_test_split)
        X_train, X_val = (slice_X(X, 0, split_at), slice_X(X, split_at))
        y_train, y_val = (y[:split_at], y[split_at:])
        return (X_train, X_val, y_train, y_val)

    def build_model(self):
        model = Sequential()
        model.add(self.model_type(len(self.char_bag), self.hidden_size))
        model.add(RepeatVector(1))
        for _ in range(self.layers):
            model.add(self.model_type(self.hidden_size, self.hidden_size,
                                      return_sequences = True))
        model.add(Dense(self.hidden_size, len(self.char_bag)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer = 'adam')
        logging.info('Model built')
        return model

    def train_model(self, model):
        for gen in range(1, self.generations):
            logging.info('Generation ' + str(gen))
            # TODO: use the whole training set each time?
            if not self.train_model_generation(model, self.verification_set()):
                logging.info('No more training data for this generation')
                break

    def sample_model(self, model, verify_x, verify_y):
        ind = np.random.randint(0, len(verify_x))
        rowX, rowY = verify_x[np.array([ind])], verify_y[np.array([ind])]
        preds = model.predict_classes(rowX, verbose = 0)
        q = self.ctable.decode(rowX[0])
        correct = self.ctable.decode(rowY[0])
        guess = self.ctable.decode(preds[0], calc_argmax = False)
        print('Q', q[::-1])
        print('T', correct)
        print((colors.ok + ' ok ' + colors.close
                   if correct == guess else colors.fail +
               ' no ' + colors.close), guess)
        print('---')

    def train_model_generation(self, model, train_test_tuple):
        train_x, verify_x, train_y, verify_y = train_test_tuple
        for k in train_test_tuple:
            if len(k) == 0:
                return False
        model.fit(train_x, train_y, batch_size = self.batch_size,
                  nb_epoch = 1, validation_data = (verify_x, verify_y),
                  show_accuracy = self.visualize_errors)
        if self.visualize_errors:
            self.sample_model(model, verify_x, verify_y)

    def train(self):
        model = self.build_model()
        self.train_model(model)
        return model


class PwdList(object):
    def __init__(self, ifile_name):
        self.ifile_name = ifile_name

    def open_file(self):
        if self.ifile_name[-3:] == '.gz':
            return gzip.open(self.ifile_name, 'rt')
        return open(self.ifile_name, 'r')

    def as_list_iter(self, agen):
        return list([row.strip('\n') for row in agen])

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
        return [row[0] for row in csv.reader(
            agen, delimiter = '\t', quotechar = None)]


class Filterer(object):
    def __init__(self, char_bag, max_len, min_len):
        self.char_bag = char_bag
        self.max_len = max_len
        self.min_len = min_len
        self.filtered_out = 0
        self.total = 0

    def signal_error(self, pwd):
        sys.stderr.write('Error, not a valid password. %s\n' % pwd)

    def pwd_is_valid(self, pwd):
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


def main(args):
    if args['log_file']:
        logging.basicConfig(filename = args['log_file'], level = logging.INFO)
    else:
        logging.basicConfig(level = logging.INFO)
    logging.info('Beginning...')
    if args['tsv']:
        input_const = TsvList
        logging.info('Input passwords in tsv format')
    else:
        input_const = PwdList
        logging.info('Input passwords in list format')
    ilist = input_const(args['ifile']).as_list()
    filt = Filterer(ModelDefaults.default_char_bag,
                    ModelDefaults.MAX_LEN, ModelDefaults.MIN_LEN)
    if args['test_input']:
        logging.info('filtering only...')
        filt.filter_test(ilist)
    else:
        logging.info('Starting training...')
        ModelSerializer(args['arch_ofile'], args['weight_ofile']).save_model(
            Trainer(filt.filter(ilist)).train())
    logging.info('Filtered %s of %s passwords', filt.filtered_out, filt.total)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Train a neural network on passwords. ')
    parser.add_argument('ifile',
                        help=('Input file name. Will be interpreted as a '
                              'gziped file if this argument ends in `.gz\'. '))
    parser.add_argument('--arch-ofile',
                        help = 'Output file for the model architecture. ',
                        default = 'model.json')
    parser.add_argument('--weight-ofile',
                        help = 'Output file for the weights of the model. ',
                        default = 'model.h5')
    parser.add_argument('--tsv', action='store_true',
                        help=('Input file is in TSV format. The first column'
                              ' of the TSV should be the password. '))
    parser.add_argument('--test-input', action='store_true', help=(
        'Test if the input is valid and print to stderr errors. Will not train'
        ' the neural network. Ignores the --ofile argument. '))
    parser.add_argument('--log-file')
    main(vars(parser.parse_args()))
