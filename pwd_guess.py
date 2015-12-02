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
import cProfile
import json

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
        model.save_weights(self.weightfile, overwrite = True)

    def load_model(self):
        with open(self.archfile, 'r') as arch:
            model = model_from_json(arch.read())
        model.load_weights(self.weightfile)
        return model

default_char_bag = (string.ascii_lowercase +
                    string.ascii_uppercase +
                    string.digits +
                    '~!@#$%^&*(),.<>/?\'"{}[]\|-_=+;:\n `')

class ModelDefaults(object):
    char_bag = default_char_bag
    hidden_size = 128
    layers = 1
    train_test_split = 0.1
    max_len = 40
    min_len = 4

    # Smaller training chunk means less memory consumed. This is especially
    # important because often, the blocking thing is using memory on the GPU
    # which is small.
    training_chunk = 128

    # More generations means it takes longer but is more accurate
    generations = 20

    visualize_num = 1
    visualize_errors = True
    model_type = recurrent.JZS1
    chunk_print_interval = 1000

    def __init__(self, adict = {}, **kwargs):
        self.adict = adict
        for k in kwargs:
            self.adict[k] = kwargs[k]

    def __getattribute__(self, name):
        if name != 'adict' and name in self.adict:
            return self.adict[name]
        else:
            return super().__getattribute__(name)

    @staticmethod
    def fromFile(afile):
        with open(afile, 'r') as f:
            return ModelDefaults(json.load(f))

class Trainer(object):
    def __init__(self, pwd_list, config = ModelDefaults()):
        self.config = config
        self.pwd_whole_list = list(pwd_list)
        self.chunk = 0
        self.ctable = self.make_char_table()

    def train_chunk_from_pwds(self, pwds):
        def all_prefixes(pwd):
            return [pwd[:i] for i in range(len(pwd))] + [pwd]
        def all_suffixes(pwd):
            return [pwd[i] for i in range(len(pwd))] + ['\n']
        def pad_to_len(astring):
            return astring + ('\n' * (self.config.max_len - len(astring)))
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

    def make_char_table(self):
        return CharacterTable(self.config.char_bag, self.config.max_len)

    def next_train_set_as_np(self):
        x_strings, y_strings = self.next_train_chunk()
        x_str_list, y_str_list = list(x_strings), list(y_strings)
        x_vec = np.zeros(
            (len(x_str_list), self.config.max_len, len(self.config.char_bag)),
            dtype = np.bool)
        for i, xstr in enumerate(x_str_list):
            x_vec[i] = self.ctable.encode(xstr, maxlen = self.config.max_len)
        y_vec = np.zeros((len(y_str_list), 1, len(self.config.char_bag)),
                         dtype = np.bool)
        for i, ystr in enumerate(y_str_list):
            y_vec[i] = self.ctable.encode(ystr, maxlen = 1)
        return (x_vec, y_vec)

    def verification_set(self, x_vec, y_vec):
        X, y = shuffle(x_vec, y_vec)
        split_at = len(X) - math.ceil(len(X) * self.config.train_test_split)
        X_train, X_val = (slice_X(X, 0, split_at), slice_X(X, split_at))
        y_train, y_val = (y[:split_at], y[split_at:])
        return (X_train, X_val, y_train, y_val)

    def build_model(self):
        model = Sequential()
        model.add(self.config.model_type(
            len(self.config.char_bag), self.config.hidden_size))
        model.add(RepeatVector(1))
        for _ in range(self.config.layers):
            model.add(self.config.model_type(
                self.config.hidden_size, self.config.hidden_size,
                return_sequences = True))
        model.add(Dense(self.config.hidden_size, len(self.config.char_bag)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer = 'adam')
        return model

    def train_model(self, model):
        for gen in range(1, self.config.generations):
            logging.info('Generation ' + str(gen))
            self.train_model_generation(model)

    def sample_model(self, model, verify_x, verify_y):
        ind = np.random.randint(0, len(verify_x))
        rowX, rowY = verify_x[np.array([ind])], verify_y[np.array([ind])]
        preds = model.predict_classes(rowX, verbose = 0)
        q = self.ctable.decode(rowX[0])
        correct = self.ctable.decode(rowY[0])
        guess = self.ctable.decode(preds[0], calc_argmax = False)
        print('Q', q.strip('\n'))
        print('T', correct)
        print((colors.ok + ' ok ' + colors.close
               if correct == guess else colors.fail +
               ' no ' + colors.close), guess)
        print('---')

    def train_model_generation(self, model):
        self.chunk = 0
        x_all, y_all = self.next_train_set_as_np()
        total_chunks = math.ceil(
            len(self.pwd_whole_list) / self.config.training_chunk)
        logging.info('Total chunks %s', total_chunks)
        while len(x_all) != 0:
            assert len(x_all) == len(y_all)
            loss, accuracy = model.train_on_batch(x_all, y_all, accuracy = True)
            if self.chunk % self.config.chunk_print_interval == 0:
                logging.info('Chunk size in cases %s', len(x_all))
                logging.info('Chunk %s of %s', self.chunk, total_chunks)
                logging.info('Loss: %s', loss)
                logging.info('Accuracy: %s', accuracy)
                if self.config.visualize_errors:
                    for _ in range(self.config.visualize_num):
                        self.sample_model(model, x_all, y_all)
            x_all, y_all = self.next_train_set_as_np()

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
    def __init__(self, config):
        self.char_bag = config.char_bag
        self.max_len = config.max_len
        self.min_len = config.min_len
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
    log_format = '%(asctime)-15s %(levelname)s: %(message)s'
    if args['log_file']:
        logging.basicConfig(filename = args['log_file'],
                            level = logging.INFO, format = log_format)
    else:
        logging.basicConfig(level = logging.INFO, format = log_format)
    logging.info('Beginning...')
    logging.info('Arguments %s', json.dumps(args, indent = 4))
    if args['tsv']:
        input_const = TsvList
        logging.info('Input passwords in tsv format')
    else:
        input_const = PwdList
        logging.info('Input passwords in list format')
    if args['config'] is None:
        config = ModelDefaults()
    else:
        config = ModelDefaults.fromFile(args['config'])
    ilist = input_const(args['ifile']).as_list()
    filt = Filterer(config)
    if args['test_input']:
        logging.info('filtering only...')
        filt.filter_test(ilist)
    else:
        logging.info('Starting training...')
        ModelSerializer(args['arch_ofile'], args['weight_ofile']).save_model(
            Trainer(filt.filter(ilist), config).train())
    logging.info('Filtered %s of %s passwords', filt.filtered_out, filt.total)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Train a neural network on passwords. ')
    parser.add_argument('ifile',
                        help=('Input file name. Will be interpreted as a '
                              'gziped file if this argument ends in `.gz\'. '))
    parser.add_argument('--arch-ofile', default = 'model.json',
                        help = 'Output file for the model architecture. ')
    parser.add_argument('--weight-ofile', default = 'model.h5',
                        help = 'Output file for the weights of the model. ')
    parser.add_argument('--tsv', action='store_true',
                        help=('Input file is in TSV format. The first column'
                              ' of the TSV should be the password. '))
    parser.add_argument('--test-input', action='store_true', help=(
        'Test if the input is valid and print to stderr errors. Will not train'
        ' the neural network. Ignores the --ofile argument. '))
    parser.add_argument('--config', help='Config file in json. ')
    parser.add_argument('--profile', help='Profile the training phase. ')
    parser.add_argument('--log-file')
    args = vars(parser.parse_args())
    main_bundle = lambda: main(args)
    if args['profile'] is not None:
        cProfile.run('main_bundle()', filename = args['profile'])
    else:
        main_bundle()
