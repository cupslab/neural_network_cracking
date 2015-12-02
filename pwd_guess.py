# -*- coding: utf-8 -*-
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
import multiprocessing
import tempfile
import subprocess as subp
import io
import collections

PASSWORD_END = '\n'

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

    def get_char_index(self, character):
        return self.char_indices[character]

    @staticmethod
    def fromConfig(config):
        if (config.uppercase_character_optimization or
            config.rare_character_optimization):
            return OptimizingCharacterTable(
                config.char_bag, config.max_len,
                config.rare_character_bag,
                config.uppercase_character_optimization)
        else:
            return CharacterTable(config.char_bag, config.max_len)

class OptimizingCharacterTable(CharacterTable):
    def __init__(self, chars, maxlen, rare_characters, uppercase):
        if uppercase:
            self.rare_characters = ''.join(
                c for c in rare_characters if c not in string.ascii_uppercase)
        else:
            self.rare_characters = rare_characters
        char_bag = chars
        for r in self.rare_characters:
            char_bag = char_bag.replace(r, '')
        char_bag += self.rare_characters[0]
        self.rare_dict = dict([(char, self.rare_characters[0])
                               for char in self.rare_characters])
        if uppercase:
            for c in string.ascii_uppercase:
                if c not in chars:
                    continue
                self.rare_dict[c] = c.lower()
                char_bag = char_bag.replace(c, '')
                assert c.lower() in char_bag
        super().__init__(char_bag, maxlen)

    def replace_all(self, astring):
        return ''.join(map(
            lambda c: self.rare_dict[c] if c in self.rare_dict else c, astring))

    def encode(self, C, maxlen = None):
        return super().encode(self.replace_all(C), maxlen)

    def get_char_index(self, character):
        if character in self.rare_dict:
            return super().get_char_index(self.rare_dict[character])
        return super().get_char_index(character)

class ModelSerializer(object):
    def __init__(self, archfile = None, weightfile = None):
        self.archfile = archfile
        self.weightfile = weightfile
        self.model_creator_from_json = model_from_json

    def save_model(self, model):
        if self.archfile is None or self.weightfile is None:
            logging.info(
                'Cannot save model because file arguments were not provided')
            return
        logging.info('Saving model architecture')
        with open(self.archfile, 'w') as arch:
            arch.write(model.to_json())
        logging.info('Saving model weights')
        model.save_weights(self.weightfile, overwrite = True)
        logging.info('Done saving model')

    def load_model(self):
        logging.info('Loading model architecture')
        with open(self.archfile, 'r') as arch:
            model = self.model_creator_from_json(arch.read())
        logging.info('Loading model weights')
        model.load_weights(self.weightfile)
        logging.info('Done loading model')
        return model

class ModelDefaults(object):
    """Configuration information for guessing and training. Can be read from a file
    in json format.

    Attributes:
    char_bag - alphabet of characters over which to guess

    model_type - type of model. Read keras documentation for more types.

    hidden_size - size of each layer. More means better accuracy

    layers - number of hidden layers. More means better accuracy

    max_len - maximum length of any password in training data. This can be
      larger than all passwords in the data and the network may output guesses
      that are this many characters long.

    min_len - minimum length of any password that will be guessed

    training_chunk - Smaller training chunk means less memory consumed. This is
      using memory on the GPU which is small. generations - More generations
      means it takes longer but is more accurate

    chunk_print_interval - Interval over which to print info to the log

    lower_probability_threshold - This controls how many passwords to output
      during generation. Lower means more passwords.

    relevel_not_matching_passwords - If true, then passwords that do not match
      the filter policy will have their probability equal to zero.

    generation_checkpoint - Every few generations, save the model.

    training_accuracy_threshold - If the accuracy is not improving by this
      amount, then quit.

    rare_character_optimization - Default false. If you specify a list of
      characters to treat as rare, then it will model those characters with a
      rare character. This will increase performance at the expense of accuracy.

    rare_character_lowest_threshold - Default 20. The with the lowest frequency
      in the training data will be modeled as special characters. This number
      indicates how many to drop.

    uppercase_character_optimization - Default false. If true, uppercase
      characters will be treated the same as lower case characters. Increases
      performance at the expense of accuracy.

    guess_serialization_method - Default is 'human'. TODO: add a compressed
      method.
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
    parallel_guessing = False
    fork_length = 2
    rare_character_optimization = False
    uppercase_character_optimization = False
    rare_character_lowest_threshold = 20
    rare_character_bag = ''
    guess_serialization_method = 'human'

    def __init__(self, adict = None, **kwargs):
        self.adict = adict if adict is not None else dict()
        for k in kwargs:
            self.adict[k] = kwargs[k]

    def __getattribute__(self, name):
        if name != 'adict' and name in self.adict:
            return self.adict[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name != 'adict':
            self.adict[name] = value
        else:
            super().__setattr__(name, value)

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
        self.model = None

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
                          len(self.ctable.chars)), dtype = np.bool)
        for i, xstr in enumerate(x_str_list):
            x_vec[i] = self.ctable.encode(xstr, maxlen = self.config.max_len)
        y_vec = np.zeros((len(y_str_list), 1, len(self.ctable.chars)),
                         dtype = np.bool)
        for i, ystr in enumerate(y_str_list):
            y_vec[i] = self.ctable.encode(ystr, maxlen = 1)
        return shuffle(x_vec, y_vec)

    def build_model(self):
        model = Sequential()
        model.add(self.config.model_type_exec()(
            len(self.ctable.chars), self.config.hidden_size))
        model.add(RepeatVector(1))
        for _ in range(self.config.layers):
            model.add(self.config.model_type_exec()(
                self.config.hidden_size, self.config.hidden_size,
                return_sequences = True))
        model.add(Dense(self.config.hidden_size, len(self.ctable.chars)))
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
        if self.model is None:
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
        self.filtered_out = 0
        self.total = 0
        self.total_characters = 0
        self.frequencies = collections.defaultdict(int)
        self.config = config

    def pwd_is_valid(self, pwd):
        pwd = pwd.strip(PASSWORD_END)
        answer = (all(map(lambda c: c in self.config.char_bag, pwd)) and
                  len(pwd) <= self.config.max_len and
                  len(pwd) >= self.config.min_len)
        if answer:
            self.total_characters += len(pwd)
            for c in pwd:
                assert c != ''
                self.frequencies[c] += 1
        else:
            self.filtered_out += 1
        self.total += 1
        return answer

    def rare_characters(self):
        lowest = list(map(
            lambda x: x[0],
            sorted(self.frequencies.items(), key = lambda x: x[1])))
        return lowest[:min(self.config.rare_character_lowest_threshold,
                           len(lowest))]

    def finish(self):
        logging.info('Filtered %s of %s passwords',
                     self.filtered_out, self.total)
        char_freqs = {}
        for key in self.frequencies:
            char_freqs[key] = self.frequencies[key] / self.total_characters
        self.config.rare_character_bag = self.rare_characters()
        logging.info('Rare characters: %s', self.config.rare_character_bag)

    def filter(self, alist):
        return filter(self.pwd_is_valid, alist)

class GuessSerializer(object):
    def __init__(self, ostream):
        self.ostream = ostream

    def serialize(self, password, prob):
        self.ostream.write('%s\t%s\n' % (
            password.strip(PASSWORD_END), prob))

    @staticmethod
    def fromConfig(config, ostream):
        if config.guess_serialization_method == 'human':
            return GuessSerializer(ostream)
        logging.warning('Unknown serialization method %s',
                        config.guess_serialization_method)
        return None

class Guesser(object):
    def __init__(self, model, config, ostream):
        self.model = model
        self.config = config
        self.generated = 0
        self.ctable = CharacterTable.fromConfig(self.config)
        self.output_serializer = GuessSerializer.fromConfig(config, ostream)
        self.filterer = Filterer(self.config)

    def cond_prob_from_preds(self, char, preds):
        return preds[self.ctable.get_char_index(char)]

    def relevel_prediction(self, preds, astring):
        if not self.filterer.pwd_is_valid(astring):
            preds[self.ctable.get_char_index(PASSWORD_END)] = 0
        elif len(astring) == self.config.max_len:
            for c in self.ctable.chars:
                preds[self.ctable.get_char_index(c)] = (
                    1 if c == PASSWORD_END else 0)
        sum_per = sum(preds)
        for i, v in enumerate(preds):
            preds[i] = v / sum_per

    def conditional_probs(self, astring):
        np_inp = np.zeros((1, self.config.max_len, len(self.ctable.chars)),
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
                self.output_serializer.serialize(astring, prob)
                self.generated += 1
            return
        prediction = self.conditional_probs(astring)
        for char in self.ctable.chars:
            chain_pass = astring + char
            chain_prob =  self.cond_prob_from_preds(char, prediction) * prob
            if (char == PASSWORD_END and
                chain_prob >= self.config.lower_probability_threshold):
                self.output_serializer.serialize(chain_pass, chain_prob)
                self.generated += 1
            elif char != PASSWORD_END:
                self.recur(chain_pass, chain_prob)

    def guess(self):
        self.recur('', 1)

    @staticmethod
    def do_guessing(model, config, ofname, start = '', start_prob = 1):
        ostream = open(ofname, 'w') if ofname != '-' else sys.stdout
        logging.info('Enumerating guesses starting at %s, %s...',
                     start, start_prob)
        guesser = Guesser(model, config, ostream)
        guesser.recur(start, start_prob)
        ostream.flush()
        ostream.close()
        logging.info('Generated %s guesses', guesser.generated)
        return guesser.generated

def fork_starting_point(arguments):
    model = ModelSerializer(*arguments['serializer']).load_model()
    return ParallelGuesser.fork_entry_point(model, arguments)

class ParallelGuesser(Guesser):
    def __init__(self, serializer, config, ostream):
        self.fork_points = []
        self.intermediate_files = []
        self.serializer = serializer
        self.forking_function = fork_starting_point
        self.tempOstream = tempfile.NamedTemporaryFile(mode = 'w')
        self.real_output = ostream
        self.fork_starter = fork_starting_point
        model = serializer.load_model()
        super().__init__(model, config, self.tempOstream)
        if self.config.fork_length > self.config.min_len:
            logging.warning(('Fork length is greater than minimum password '
                             'length. This may cause some passwords that should'
                             ' be guessed to not show up in output. '))

    def recur(self, astring, prob):
        if len(astring) == self.config.fork_length:
            self.fork_points.append((astring, prob))
        else:
            super().recur(astring, prob)

    def guess(self):
        self.recur('', 1)
        self.do_forking()

    def prepare_argument_dict(self, node):
        new_file = tempfile.NamedTemporaryFile()
        self.intermediate_files.append(new_file)
        return {
            'config' : self.config.as_dict(),
            'serializer' : [
                self.serializer.archfile, self.serializer.weightfile],
            'node' : node,
            'ofile' : new_file.name
        }

    def arg_list(self):
        return list(map(self.prepare_argument_dict, self.fork_points))

    def collect_answer(self):
        streams = list(map(lambda n: open(n.name, 'r'),
                           [self.tempOstream] + self.intermediate_files))
        for stream in streams:
            for line in stream:
                self.real_output.write(line)
            stream.close()
        self.real_output.flush()

    def do_forking(self):
        arg_list = self.arg_list()
        pool = multiprocessing.Pool(
            min(len(arg_list), multiprocessing.cpu_count()))
        result = pool.map_async(self.fork_starter, arg_list)
        try:
            pool.close()
            pool.join()
            answer = result.get(timeout = 1)
            self.generated = sum(answer) + self.generated
            self.collect_answer()
        except KeyboardInterrupt as e:
            logging.error('Received keyboard interrupt. Stopping processes...')
            pool.terminate()

    @staticmethod
    def fork_entry_point(model, arguments):
        config = ModelDefaults(**arguments['config'])
        return Guesser.do_guessing(model, config, arguments['ofile'],
                                   arguments['node'][0], arguments['node'][1])

    @staticmethod
    def do_guessing(serializer, config, ofname):
        ostream = open(ofname, 'w') if ofname != '-' else sys.stdout
        logging.info('Enumerating guesses...')
        guesser = ParallelGuesser(serializer, config, ostream)
        guesser.guess()
        logging.info('Generated %s guesses', guesser.generated)
        return guesser.generated

log_level_map = {
    'info' : logging.INFO,
    'warning'  : logging.WARNING,
    'debug' : logging.DEBUG,
    'error' : logging.ERROR
}

def get_version_string():
    p = subp.Popen(['git', 'describe'],
                   stdin=subp.PIPE, stdout=subp.PIPE, stderr=subp.PIPE)
    output, err = p.communicate()
    return output.decode('utf-8').strip('\n')

def init_logging(args, config):
    log_format = '%(asctime)-15s %(levelname)s: %(message)s'
    log_level = log_level_map[args['log_level']]
    if args['log_file']:
        logging.basicConfig(filename = args['log_file'],
                            level = log_level, format = log_format)
    else:
        logging.basicConfig(level = log_level, format = log_format)
    logging.info('Beginning...')
    logging.info('Arguments: %s', json.dumps(args, indent = 4))
    logging.info('Configuration: %s', json.dumps(config.as_dict(), indent = 4))
    logging.info('Version: %s', get_version_string())
    def except_hook(exctype, value, tb):
        logging.critical('Uncaught exception', exc_info = (exctype, value, tb))
        sys.stderr.write('Uncaught exception!')
    sys.excepthook = except_hook

def train(args, config):
    if args['tsv']:
        input_const = TsvList
    else:
        input_const = PwdList
    filt = Filterer(config)
    logging.info('Reading training set...')
    plist = list(filt.filter(input_const(args['pwd_file']).as_list()))
    filt.finish()
    logging.info('Done reading passwords...')
    if len(plist) == 0:
        logging.error('Empty training set! Quiting...')
        sys.exit(1)
    if args['pre_processing_only']:
        logging.info('Only performing pre-processing. ')
        sys.exit(0)
    trainer = Trainer(plist, config)
    serializer = ModelSerializer(args['arch_file'], args['weight_file'])
    if args['retrain']:
        logging.info('Retraining model...')
        trainer.model = serializer.load_model()
    trainer.train(serializer)

def guess(args, config):
    logging.info('Loading model...')
    if args['arch_file'] is None or args['weight_file'] is None:
        logging.error('Architecture file or weight file not found. Quiting...')
        sys.exit(1)
    serializer = ModelSerializer(args['arch_file'], args['weight_file'])
    if config.parallel_guessing:
        ParallelGuesser.do_guessing(
            serializer, config, args['enumerate_ofile'])
    else:
        Guesser.do_guessing(
            serializer.load_model(), config, args['enumerate_ofile'])

def main(args):
    if args['help_config']:
        sys.stdout.write(ModelDefaults.__doc__ + '\n')
        sys.exit(0)
    if args['version']:
        sys.stdout.write(get_version_string() + '\n')
        sys.exit(0)
    config = ModelDefaults.fromFile(args['config'])
    init_logging(args, config)
    if args['pwd_file']:
        train(args, config)
    if args['enumerate_ofile']:
        guess(args, config)
    if not args['pwd_file'] and not args['enumerate_ofile']:
        logging.error('Nothing to do! Use --pwd-file or --enumerate-ofile. ')
        sys.exit(1)
    logging.info('Done!')

def make_parser():
    parser = argparse.ArgumentParser(description=(
        """Neural Network with passwords. This program uses a neural network to
        guess passwords. This happens in two phases, training and
        enumeration. Either --pwd-file or --enumerate-ofile are required.
        --pwd-file will give a password file as training data.
        --enumerate-ofile will guess passwords based on an existing model.
        Version """ +
        get_version_string()))
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
                              ' password. Second column is the frequency count '
                              'of the password'))
    parser.add_argument('--enumerate-ofile',
                        help = 'Enumerate guesses output file')
    parser.add_argument('--retrain', action='store_true',
                        help = ('Instead of training a new model, begin '
                                'training the model in the weight-file and '
                                'arch-file arguments. '))
    parser.add_argument('--config', help = 'Config file in json. ')
    parser.add_argument('--profile',
                        help = 'Profile execution and save to the given file. ')
    parser.add_argument('--help-config', action = 'store_true',
                        help = 'Print help for config files and exit')
    parser.add_argument('--log-file')
    parser.add_argument('--log-level', default = 'info',
                        choices = ['debug', 'info', 'warning', 'error'])
    parser.add_argument('--version', action = 'store_true',
                        help = 'Print version number and exit')
    parser.add_argument('--pre-processing-only', action='store_true',
                        help = 'Only perform the preprocessing step. ')
    return parser

if __name__=='__main__':
    args = vars(make_parser().parse_args())
    main_bundle = lambda: main(args)
    if args['profile'] is not None:
        cProfile.run('main_bundle()', filename = args['profile'])
    else:
        main_bundle()
