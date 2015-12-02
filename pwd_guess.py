# -*- coding: utf-8 -*-
# William Melicher
from __future__ import print_function
from keras.models import Sequential, slice_X, model_from_json
from keras.layers.core import Activation, Dense, RepeatVector
from keras.layers import recurrent
from keras.optimizers import SGD
from sklearn.utils import shuffle
import numpy as np
from sqlitedict import SqliteDict

import sys
import argparse
import itertools
import string
import gzip
import csv
import logging
import cProfile
import json
import random
import multiprocessing
import tempfile
import subprocess as subp
import collections
import struct
import os.path

PASSWORD_END = '\n'
PASSWORD_FRAGMENT_DELIMITER = '\0'

class BaseTrie(object):
    def increment(self, aword, weight = 1):
        raise NotImplementedError()

    def iterate(self, serial_type):
        raise NotImplementedError()

    def finish(self):
        pass

    config_keys = {
        'trie' : lambda _: NodeTrie(),
        'disk' : lambda c: DiskBackedTrie(c),
        None : lambda _: BaseTrie()
    }

    @staticmethod
    def fromConfig(config):
        if config.trie_implementation in BaseTrie.config_keys:
            return BaseTrie.config_keys[config.trie_implementation](config)
        logging.error('Unknown trie type %s. ', config.trie_implementation)

class NodeTrie(BaseTrie):
    def __init__(self):
        self.nodes = collections.defaultdict(NodeTrie)
        self.weight = 0
        self._size = 0

    def size(self):
        return self._size

    def increment(self, aword, weight = 1):
        self.weight += weight
        answer = 0
        if len(aword) != 0:
            if aword[0] not in self.nodes:
                answer += 1
            returned = self.nodes[sys.intern(aword[0])].increment(
                sys.intern(aword[1:]), weight)
            self._size += returned + answer
            return returned + answer
        return answer

    def random_iterate(self, cur = ''):
        if cur != '':
            yield (cur, self.weight)
        for key in self.nodes:
            others = self.nodes[key].random_iterate(cur + key)
            for item in others:
                yield item

    def sampled_training(self, value = ''):
        node_children = list(map(lambda k: (k, self.nodes[k].weight),
                                 sorted(self.nodes.keys())))
        if len(node_children) == 0:
            return
        yield (value, node_children)
        for key in self.nodes:
            for item in self.nodes[key].sampled_training(value + key):
                yield item

    def iterate(self, serial_type):
        if serial_type == 'fuzzy':
            return self.sampled_training()
        else:
            return self.random_iterate()

class DiskBackedTrie(BaseTrie):
    def __init__(self, config):
        self.config = config
        self.current_node = None
        self.current_branch_key = None
        self.weights = NodeTrie()
        self.keys = []

    def finish(self):
        self.close_branch()
        self.config.set_intermediate_info('db_trie_keys', self.keys)
        self.config.set_intermediate_info('db_trie_weights', self.weights)
        logging.info('Finishing disk backed trie iteration')

    def make_serializer(self):
        return TrieSerializer.getFactory(self.config, True)(
            os.path.join(self.config.trie_intermediate_storage,
                         self.sanitize(self.current_branch_key)),
            self.config.max_len,
            self.config.trie_serializer_encoding)

    def sanitize(self, prefix):
        assert prefix in self.keys
        return str(self.keys.index(prefix))

    def close_branch(self):
        if self.current_branch_key is not None:
            assert self.current_node is not None
            self.make_serializer().serialize(self.current_node)
        self.current_node = None
        self.current_branch_key = None

    def open_new_branch(self, key):
        self.close_branch()
        self.current_node = NodeTrie()
        self.current_branch_key = key
        self.keys.append(key)

    def increment(self, aword, weight = 1):
        start, end = (aword[:self.config.fork_length],
                      aword[self.config.fork_length:])
        if start != self.current_branch_key:
            self.open_new_branch(start)
        self.weights.increment(start, weight)
        self.current_node.increment(end, weight)

    def iterate_subtrees(self, serial_type):
        for key in self.keys:
            self.current_branch_key = key
            for subitem in self.make_serializer().deserialize():
                yield (key + subitem[0], subitem[1])

    def iterate(self, serial_type):
        self.finish()
        for c in self.weights.iterate(serial_type):
            yield c
        for s in self.iterate_subtrees(serial_type):
            yield s
        self.current_branch_key = None

    @classmethod
    def fromIntermediate(cls, config):
        answer = cls(config)
        answer.keys = config.get_intermediate_info('db_trie_keys')
        answer.weights = config.get_intermediate_info('db_trie_weights')
        return answer

class TrieSerializer(object):
    def __init__(self, fname, max_len, encoding):
        self.fname = fname
        self.max_len = max_len
        self.encoding = encoding

    def open_file(self, mode):
        return gzip.open(self.fname, mode)

    def serialize(self, trie):
        directory = os.path.dirname(self.fname)
        if not os.path.exists(directory) and directory != '':
            logging.info('Making directory to save %s', directory)
            os.mkdir(directory)
        self.do_serialize(trie)

    def do_serialize(self, trie):
        raise NotImplementedError()

    def deserialize(self):
        raise NotImplementedError()

    @staticmethod
    def fromConfig(config):
        return TrieSerializer.getFactory(config)(
            config.trie_fname, config.max_len, config.trie_serializer_encoding)

    @staticmethod
    def getFactory(config, intermediate_serializer = False):
        if config.trie_fname == ':memory:' and not intermediate_serializer:
            return lambda x, y, z: MemoryTrieSerializer(
                config.trie_serializer_type, x, y, z)
        elif (config.trie_intermediate_storage == ':memory:'
              and intermediate_serializer):
            return lambda x, y, z: MemoryTrieSerializer(
                config.trie_serializer_type, x, y, z)
        elif config.trie_serializer_type == 'fuzzy':
            return TrieFuzzySerializer
        elif config.trie_serializer_type == 'reg':
            return NodeTrieSerializer
        logging.error('No serializer of type %s', config.trie_serializer_type)

class MemoryTrieSerializer(TrieSerializer):
    memory_cache = {}

    def __init__(self, serializer_type, *args):
        super().__init__(*args)
        self.serializer_type = serializer_type

    def serialize(self, trie):
        self.memory_cache[self.fname] = trie

    def deserialize(self):
        trie = self.memory_cache[self.fname]
        return trie.iterate(self.serializer_type)

class NodeTrieSerializer(TrieSerializer):
    def __init__(self, *args):
        super().__init__(*args)
        self.fmt = '<' + str(self.max_len) + 'sQ'
        self.chunk_size = struct.calcsize(self.fmt)

    def do_serialize(self, trie):
        with self.open_file('wb') as afile:
            for item in trie.iterate('reg'):
                pwd, weight = item
                afile.write(struct.pack(
                    self.fmt, pwd.encode(self.encoding), weight))

    def load_one_record(self, somebytes):
        pwd, weight =  struct.unpack_from(self.fmt, somebytes)
        return (pwd.decode(self.encoding)
                .strip(PASSWORD_FRAGMENT_DELIMITER),
                weight)

    def deserialize(self):
        with self.open_file('rb') as afile:
            while True:
                chunk = afile.read(self.chunk_size)
                if chunk:
                    yield self.load_one_record(chunk)
                else:
                    break

class TrieFuzzySerializer(TrieSerializer):
    def __init__(self, *args):
        super().__init__(*args)
        self.in_fmt = '<' + str(self.max_len) + 'sH'
        self.out_fmt = '<1sQ'
        self.in_fmt_bytes = struct.calcsize(self.in_fmt)
        self.out_fmt_bytes = struct.calcsize(self.out_fmt)

    def write_record(self, ostream, input_str, output_list):
        ostream.write(struct.pack(
            self.in_fmt, input_str.encode(self.encoding), len(output_list)))
        for item in output_list:
            char, weight = item
            ostream.write(struct.pack(
                self.out_fmt, char.encode(self.encoding), weight))

    def read_record(self, istream):
        input_rec = istream.read(self.in_fmt_bytes)
        if len(input_rec) == 0:
            return None
        pwd_bytes, num_rec = struct.unpack_from(self.in_fmt, input_rec)
        record = []
        for _ in range(num_rec):
            out_char, out_weight = struct.unpack_from(
                self.out_fmt, istream.read(self.out_fmt_bytes))
            record.append(
                (out_char.decode(self.encoding), out_weight))
        return (pwd_bytes.decode(self.encoding)
                .strip(PASSWORD_FRAGMENT_DELIMITER), record)

    def do_serialize(self, trie):
        with self.open_file('wb') as afile:
            for item in trie.iterate('fuzzy'):
                pwd, records = item
                self.write_record(afile, pwd, records)

    def deserialize(self):
        with self.open_file('rb') as afile:
            while True:
                answer = self.read_record(afile)
                if answer is None:
                    break
                yield answer

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

    def pad_to_len(self, astring, maxlen = None):
        maxlen = maxlen if maxlen else self.maxlen
        return astring + (PASSWORD_END * (maxlen - len(astring)))

    def encode_many(self, string_list, maxlen = None):
        maxlen = maxlen if maxlen else self.maxlen
        x_str_list = list(map(
            lambda x: self.pad_to_len(x, maxlen), string_list))
        x_vec = np.zeros((len(x_str_list), maxlen, len(self.chars)),
                         dtype = np.bool)
        for i, xstr in enumerate(x_str_list):
            x_vec[i] = self.encode(xstr, maxlen = maxlen)
        return x_vec

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
                config.get_intermediate_info('rare_character_bag'),
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
    char_bag = (string.ascii_lowercase + string.ascii_uppercase +
                string.digits + '~!@#$%^&*(),.<>/?\'"{}[]\|-_=+;: `' +
                PASSWORD_END)
    model_type = 'JZS1'
    hidden_size = 128
    layers = 1
    max_len = 40
    min_len = 4
    training_chunk = 128
    generations = 20
    chunk_print_interval = 1000
    lower_probability_threshold = 10**-5
    relevel_not_matching_passwords = True
    training_accuracy_threshold = 10**-10
    train_test_ratio = 10
    parallel_guessing = False
    fork_length = 2
    rare_character_optimization = False
    uppercase_character_optimization = False
    rare_character_lowest_threshold = 20
    guess_serialization_method = 'human'
    simulated_frequency_optimization = False
    trie_implementation = None
    trie_fname = ':memory:'
    trie_intermediate_storage = ':memory:'
    intermediate_fname = 'intermediate_fname.sqlite'
    trie_serializer_encoding = 'utf8'
    trie_serializer_type = 'reg'
    save_always = True
    randomize_training_order = True

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
                answer = ModelDefaults(json.load(f))
            except ValueError as e:
                logging.error(('Error loading config. Config file is not valid'
                               ' JSON format. %s'), str(e))
                return None
        if answer.trie_serializer_type == 'fuzzy':
            assert answer.simulated_frequency_optimization
            assert answer.trie_implementation is not None
        assert answer.fork_length < answer.min_len
        return answer

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

    def set_intermediate_info(self, key, value):
        with SqliteDict(self.intermediate_fname) as info:
            info[key] = value
            info.commit()

    def get_intermediate_info(self, key):
        try:
            with SqliteDict(self.intermediate_fname) as info:
                return info[key]
        except KeyError as e:
            logging.error('Cannot find intermediate data %s. Looking in %s',
                          str(e), self.intermediate_fname)
            raise

class BasePreprocessor(object):
    def __init__(self, config = ModelDefaults()):
        self.config = config

    def begin(self, anobj):
        raise NotImplementedError()

    def next_chunk(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def stats(self):
        self.reset()
        x_vec, y_vec, weights = self.next_chunk()
        count_instances = 0
        while len(x_vec) != 0:
            count_instances += len(x_vec)
            x_vec, y_vec, weights = self.next_chunk()
        logging.info('Number of training instances %s', count_instances)
        return count_instances

    config_keys = {
        'trie' : lambda c: TriePreprocessor(c),
        'disk' : lambda c: HybridDiskPreprocessor(c),
        None : lambda c: Preprocessor(c)
    }

    format_keys = {
        'trie' : lambda c: DiskPreprocessor(c),
        'im_trie' : lambda c: IntermediatePreprocessor(c)
    }

    @staticmethod
    def fromConfig(config):
        logging.info('Preprocessor type %s...', config.trie_implementation)
        if config.trie_implementation in BasePreprocessor.config_keys:
            return BasePreprocessor.config_keys[
                config.trie_implementation](config)
        logging.error('Cannot find trie_implementation %s',
                      config.trie_implementation)

    @staticmethod
    def byFormat(pwd_format, config):
        if pwd_format in BasePreprocessor.format_keys:
            return BasePreprocessor.format_keys[pwd_format](config)
        logging.error('Cannot find preprocessor from format %s', pwd_format)

class Preprocessor(BasePreprocessor):
    def __init__(self, config = ModelDefaults()):
        super().__init__(config)
        self.chunk = 0

    def begin(self, pwd_list):
        self.pwd_whole_list = list(pwd_list)
        self.reset()

    def all_prefixes(self, pwd):
        return [pwd[:i] for i in range(len(pwd))] + [pwd]

    def all_suffixes(self, pwd):
        return [pwd[i] for i in range(len(pwd))] + [PASSWORD_END]

    def repeat_weight(self, pwd):
        return [self.password_weight(pwd) for _ in range(len(pwd) + 1)]

    def train_from_pwds(self, pwd_tuples):
        self.pwd_freqs = dict(pwd_tuples)
        pwds = list(map(lambda x: x[0], pwd_tuples))
        return (
            itertools.chain.from_iterable(map(self.all_prefixes, pwds)),
            itertools.chain.from_iterable(map(self.all_suffixes, pwds)),
            itertools.chain.from_iterable(map(self.repeat_weight, pwds)))

    def next_chunk(self):
        if self.chunk * self.config.training_chunk >= len(self.pwd_whole_list):
            return [], [], []
        pwd_list = self.pwd_whole_list[
            self.chunk * self.config.training_chunk:
            min((self.chunk + 1) * self.config.training_chunk,
                len(self.pwd_whole_list))]
        self.chunk += 1
        pwd_input, output, weight = self.train_from_pwds(pwd_list)
        return (
            list(pwd_input), list(output), list(weight))

    def password_weight(self, pwd):
        if pwd in self.pwd_freqs:
            return self.pwd_freqs[pwd]
        logging.warning('Cannot find frequency for password %s', pwd)
        return 1

    def reset(self):
        self.chunk = 0
        if self.config.randomize_training_order:
            random.shuffle(self.pwd_whole_list)

class TriePreprocessor(BasePreprocessor):
    def __init__(self, config = ModelDefaults()):
        super().__init__(config)
        self.instances = 0
        self.trie = BaseTrie.fromConfig(config)

    def preprocess(self, pwd_list):
        return pwd_list

    def begin(self, pwd_list):
        out_pwd_list = self.preprocess(pwd_list)
        for item in out_pwd_list:
            pwd, weight = item
            self.instances += len(pwd) + 1
            self.trie.increment(pwd + PASSWORD_END, weight)
        self.trie.finish()
        logging.info('Saving preprocessing step...')
        TrieSerializer.fromConfig(self.config).serialize(self.trie)

    def compress_list(self, x, y, w):
        for i in range(len(x)):
            self.trie.increment(x[i] + y[i], w[i])
            self.instances += 1

    def reset(self):
        self.set_iterator()
        if self.config.randomize_training_order:
            self.current_generator = list(self.current_generator)
            random.shuffle(self.current_generator)
            self.current_generator = iter(self.current_generator)

    def set_iterator(self):
        self.current_generator = self.trie.iterate(
            self.config.trie_serializer_type)

    def next_chunk(self):
        x, y, w = [], [], []
        for item in list(itertools.islice(
                self.current_generator, self.config.training_chunk)):
            key, value = item
            if type(value) == list:
                x.append(key)
                y.append(value)
                w.append(1)
            else:
                x.append(key[:-1])
                y.append(key[-1])
                w.append(value)
        return (x, y, w)

class DiskPreprocessor(TriePreprocessor):
    def begin(self, pwd_file = None):
        self.serializer = TrieSerializer.getFactory(self.config)(
            pwd_file if pwd_file is not None else self.config.trie_fname,
            self.config.max_len, self.config.trie_serializer_encoding)

    def set_iterator(self):
        self.current_generator = self.serializer.deserialize()

class IntermediatePreprocessor(TriePreprocessor):
    def begin(self, pwd_file = None):
        logging.info('Loading trie intermediate representation...')
        self.trie = DiskBackedTrie.fromIntermediate(self.config)

class HybridDiskPreprocessor(TriePreprocessor):
    def preprocess(self, pwd_list):
        # TODO: make memory efficient
        out_pwd_list = collections.defaultdict(list)
        for item in pwd_list:
            out_pwd_list[item[0][:self.config.fork_length]].append(item)
        return self.read(out_pwd_list)

    def read(self, pwd_list):
        for key in sorted(pwd_list.keys()):
            subkeys = pwd_list[key]
            for item in subkeys:
                yield item

class Trainer(object):
    def __init__(self, pwd_list, config = ModelDefaults()):
        self.config = config
        self.chunk = 0
        self.generation = 0
        self.ctable = CharacterTable.fromConfig(self.config)
        self.model = None
        self.pwd_list = pwd_list

    def next_train_set_as_np(self):
        x_strs, y_str_list, weight_list = self.pwd_list.next_chunk()
        x_vec = self.ctable.encode_many(x_strs)
        y_vec = self.prepare_y_data(y_str_list)
        weight_vec = np.zeros((len(weight_list), 1, 1))
        for i, weight in enumerate(weight_list):
            weight_vec[i] = weight
        return shuffle(x_vec, y_vec, weight_vec)

    def prepare_y_data(self, y_str_list):
        y_vec = np.zeros((len(y_str_list), 1, len(self.ctable.chars)),
                         dtype = np.bool)
        for i, ystr in enumerate(y_str_list):
            y_vec[i] = self.ctable.encode(ystr, maxlen = 1)
        return y_vec

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
        max_accuracy = 0
        for gen in range(self.config.generations):
            self.generation = gen + 1
            logging.info('Generation ' + str(gen + 1))
            accuracy = self.train_model_generation()
            logging.info('Generation accuracy: %s', accuracy)
            if accuracy > max_accuracy or self.config.save_always:
                max_accuracy = accuracy
                serializer.save_model(self.model)
            if ((accuracy - prev_accuracy) <
                self.config.training_accuracy_threshold):
                logging.info('Accuracy diff of %s is less than threshold.',
                             accuracy - prev_accuracy)
                break
            prev_accuracy = accuracy

    def test_set(self, x_all, y_all, w_all):
        split_at = len(x_all) - max(
            int(len(x_all) / self.config.train_test_ratio), 1)
        x_train, x_val = (slice_X(x_all, 0, split_at), slice_X(x_all, split_at))
        y_train, y_val = (y_all[:split_at], y_all[split_at:])
        w_train, w_val = (w_all[:split_at], w_all[split_at:])
        return x_train, x_val, y_train, y_val, w_train, w_val

    def training_step(self, x_all, y_all, w_all):
        x_train, x_val, y_train, y_val, w_train, w_val = self.test_set(
            x_all, y_all, w_all)
        train_loss, train_accuracy = self.model.train_on_batch(
            x_train, y_train, accuracy = True, sample_weight = w_train)
        test_loss, test_accuracy = self.model.test_on_batch(
            x_val, y_val, accuracy = True, sample_weight = w_val)
        return train_loss, train_accuracy, test_loss, test_accuracy

    def train_model_generation(self):
        self.chunk = 0
        self.pwd_list.reset()
        accuracy_accum = []
        x_all, y_all, w_all = self.next_train_set_as_np()
        chunk = 0
        while len(x_all) != 0:
            assert len(x_all) == len(y_all)
            tr_loss, _, te_loss, te_acc = self.training_step(
                x_all, y_all, w_all)
            accuracy_accum += [(len(x_all), te_acc)]
            if chunk % self.config.chunk_print_interval == 0:
                logging.info('Chunk %s. Each chunk is size %s',
                             chunk, len(x_all))
                logging.info('Train loss %s. Test loss %s. Test accuracy %s.',
                             tr_loss, te_loss, te_acc)
            x_all, y_all, w_all = self.next_train_set_as_np()
            chunk += 1
        instances = map(lambda x: x[0], accuracy_accum)
        logging.info('Trained on %s instances this generation', instances)
        return sum(map(lambda x: x[0] * x[1], accuracy_accum)) / sum(
            instances)

    def train(self, serializer):
        logging.info('Building model...')
        if self.model is None:
            self.build_model()
        logging.info('Done compiling model. Beginning training...')
        self.train_model(serializer)

    @staticmethod
    def getFactory(config):
        if config.trie_serializer_type == 'fuzzy':
            logging.info('Fuzzy trie trainer')
            return FuzzyTrieTrainer
        else:
            logging.info('Regular trainer')
            return Trainer

class FuzzyTrieTrainer(Trainer):
    def prepare_y_data(self, y_str_list):
        y_vec = np.zeros((len(y_str_list), 1, len(self.ctable.chars)))
        for i, records in enumerate(y_str_list):
            value = np.zeros((1, len(self.ctable.chars)))
            weight_sum = 0
            for record in records:
                outchar, weight = record
                weight_sum += weight
                value += self.ctable.encode(outchar, maxlen = 1) * weight
            y_vec[i] = value / weight_sum
        return y_vec

class PwdList(object):
    def __init__(self, ifile_name):
        self.ifile_name = ifile_name

    def open_file(self):
        if self.ifile_name[-3:] == '.gz':
            return gzip.open(self.ifile_name, 'rt')
        return open(self.ifile_name, 'r')

    def as_list_iter(self, agen):
        return [(row.strip(PASSWORD_END), 1) for row in agen]

    def as_list(self):
        answer = []
        ifile = self.open_file()
        try:
            answer = self.as_list_iter(ifile)
        finally:
            ifile.close()
        return answer

    @staticmethod
    def getFactory(file_format, config):
        if file_format == 'tsv':
            if config.simulated_frequency_optimization:
                return TsvSimulatedList
            else:
                return TsvList
        elif file_format == 'list':
            if config.simulated_frequency_optimization:
                logging.error('Cannot enable simulated_frequency_optimization'
                              ' on list format. Must be in TSV format. ')
            return PwdList
        logging.error('Cannot find factory for format of %s', file_format)

class TsvList(PwdList):
    def as_list_iter(self, agen):
        answer = []
        for row in csv.reader(agen, delimiter = '\t', quotechar = None):
            pwd, freq = row[:2]
            for _ in range(int(float.fromhex(freq))):
                answer.append((sys.intern(pwd), 1))
        return answer

class TsvSimulatedList(PwdList):
    def as_list_iter(self, agen):
        return [(row[0], int(float.fromhex(row[1]))) for row in
                csv.reader(agen, delimiter = '\t', quotechar = None)]

class Filterer(object):
    def __init__(self, config):
        self.filtered_out = 0
        self.total = 0
        self.total_characters = 0
        self.frequencies = collections.defaultdict(int)
        self.config = config
        self.longest_pwd = 0

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
        self.longest_pwd = max(self.longest_pwd, len(pwd))
        return answer

    def rare_characters(self):
        lowest = list(map(
            lambda x: x[0],
            sorted(map(lambda c: (c, self.frequencies[c]),
                       self.config.char_bag.replace(PASSWORD_END, '')),
                   key = lambda x: x[1])))
        return lowest[:min(self.config.rare_character_lowest_threshold,
                           len(lowest))]

    def finish(self):
        logging.info('Filtered %s of %s passwords',
                     self.filtered_out, self.total)
        char_freqs = {}
        for key in self.frequencies:
            char_freqs[key] = self.frequencies[key] / self.total_characters
        if self.config.rare_character_optimization:
            self.config.set_intermediate_info(
                'rare_character_bag', self.rare_characters())
        logging.info('Rare characters: %s', self.rare_characters())
        logging.info('Longest pwd is : %s characters long', self.longest_pwd)
        self.config.max_len = self.longest_pwd

    def filter(self, alist):
        return filter(lambda x: self.pwd_is_valid(x[0]), alist)

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
        self.tempOstream = tempfile.NamedTemporaryFile(mode = 'w')
        model = serializer.load_model()
        super().__init__(model, config, self.tempOstream)
        self.fork_points = []
        self.intermediate_files = []
        self.serializer = serializer
        self.forking_function = fork_starting_point
        self.real_output = ostream
        self.fork_starter = fork_starting_point

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
                   cwd = os.path.dirname(os.path.realpath(__file__)),
                   stdin=subp.PIPE, stdout=subp.PIPE, stderr=subp.PIPE)
    output, err = p.communicate()
    return output.decode('utf-8').strip('\n')

def init_logging(args):
    def except_hook(exctype, value, tb):
        logging.critical('Uncaught exception', exc_info = (exctype, value, tb))
        sys.stderr.write('Uncaught exception! See log for details.\n')
    sys.excepthook = except_hook
    config = ModelDefaults.fromFile(args['config'])
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
    return config

def read_passwords(pwd_file, pwd_format, config):
    input_factory = PwdList.getFactory(pwd_format, config)
    filt = Filterer(config)
    logging.info('Reading training set...')
    plist = list(filt.filter(input_factory(pwd_file).as_list()))
    filt.finish()
    logging.info('Done reading passwords...')
    return plist

def preprocessing(args, config):
    if args['pwd_format'] in BasePreprocessor.format_keys:
        logging.info('Formatting preprocessor')
        disk_trie = BasePreprocessor.byFormat(args['pwd_format'], config)
        disk_trie.begin(args['pwd_file'])
        return disk_trie
    preprocessor = BasePreprocessor.fromConfig(config)
    preprocessor.begin(
        read_passwords(args['pwd_file'], args['pwd_format'], config))
    if args['pre_processing_only']:
        logging.info('Only performing pre-processing. ')
        preprocessor.stats()
        return None
    return preprocessor

def train(args, config):
    preprocessor = preprocessing(args, config)
    if preprocessor is None:
        return
    trainer = (Trainer.getFactory(config))(preprocessor, config)
    serializer = ModelSerializer(args['arch_file'], args['weight_file'])
    if args['retrain']:
        logging.info('Retraining model...')
        trainer.model = serializer.load_model()
    trainer.train(serializer)
    if args['enumerate_ofile']:
        if config.parallel_guessing:
            ParallelGuesser.do_guessing(
                serializer, config, args['enumerate_ofile'])
        else:
            Guesser.do_guessing(trainer.model, config, args['enumerate_ofile'])

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
    if args['version']:
        sys.stdout.write(get_version_string() + '\n')
        sys.exit(0)
    config = init_logging(args)
    if args['pwd_file']:
        train(args, config)
    elif args['enumerate_ofile']:
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
    parser.add_argument('--pwd-file', help=('Input file name. '))
    parser.add_argument('--arch-file',
                        help = 'Output file for the model architecture. ')
    parser.add_argument('--weight-file',
                        help = 'Output file for the weights of the model. ')
    parser.add_argument('--pwd-format', default = 'list',
                        choices = ['trie', 'tsv', 'list', 'im_trie'],
                        help = ('Format of pwd-file input. "list" format is one'
                                'password per line. "tsv" format is tab '
                                'separated values: first column is the '
                                'password, second is the frequency in floating'
                                ' hex. "trie" is a custom binary format created'
                                ' by another step of this tool. '))
    parser.add_argument('--enumerate-ofile',
                        help = 'Enumerate guesses output file')
    parser.add_argument('--retrain', action='store_true',
                        help = ('Instead of training a new model, begin '
                                'training the model in the weight-file and '
                                'arch-file arguments. '))
    parser.add_argument('--config', help = 'Config file in json. ')
    parser.add_argument('--profile',
                        help = 'Profile execution and save to the given file. ')
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
