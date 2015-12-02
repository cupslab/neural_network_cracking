# -*- coding: utf-8 -*-
# author: William Melicher
from __future__ import print_function
from keras.models import Sequential, slice_X, model_from_json
from keras.layers.core import Activation, Dense, RepeatVector, TimeDistributedDense
from keras.layers import recurrent
from keras.optimizers import SGD
from sklearn.utils import shuffle
import numpy as np
from sqlitedict import SqliteDict
import theano

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
import multiprocessing as mp
import tempfile
import subprocess as subp
import collections
import struct
import os.path
import mmap
import bisect

PASSWORD_END = '\n'

FNAME_PREFIX_PREPROCESSOR = 'disk_cache.'
FNAME_PREFIX_TRIE = 'trie_nodes.'

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
        try:
            return BaseTrie.config_keys[config.trie_implementation](config)
        except KeyError as e:
            logging.error('Cannot find trie type %s.',
                          config.trie_implementation)

class NodeTrie(BaseTrie):
    def __init__(self):
        self.nodes = collections.defaultdict(NodeTrie)
        self.weight = 0
        self._size = 0

    @staticmethod
    def increment_optimized(anode, aword, weight = 1):
        root = anode
        inc_str = aword
        root.weight += weight
        while len(inc_str) != 0:
            next_char, inc_str = inc_str[0], inc_str[1:]
            root.weight += weight
            root = root.nodes[next_char]
        root.weight += weight

    def increment(self, aword, weight = 1):
        NodeTrie.increment_optimized(self, aword, weight)

    def random_iterate(self, cur = ''):
        if cur != '':
            yield (cur, self.weight)
        for key in self.nodes:
            others = self.nodes[key].random_iterate(cur + key)
            for item in others:
                yield item

    def sampled_training(self, value = ''):
        node_children = [(k, self.nodes[k].weight)
                         for k in sorted(self.nodes.keys())]
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
        self.fork_length = config.fork_length

    def finish(self):
        if self.current_branch_key is not None:
            self.close_branch()
            self.config.set_intermediate_info('db_trie_keys', self.keys)
            self.config.set_intermediate_info('db_trie_weights', self.weights)
            logging.info('Finishing disk backed trie iteration')

    def make_serializer(self):
        return TrieSerializer.getFactory(self.config, True)(
            os.path.join(self.config.trie_intermediate_storage,
                         self.sanitize(self.current_branch_key)))

    def sanitize(self, prefix):
        assert prefix in self.keys
        return FNAME_PREFIX_TRIE + str(self.keys.index(prefix))

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
        start, end = (aword[:self.fork_length], aword[self.fork_length:])
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
    def __init__(self, fname):
        self.fname = fname

    def open_file(self, mode, fname = None):
        return open(fname if fname else self.fname, mode)

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
        return TrieSerializer.getFactory(config)(config.trie_fname)

    @staticmethod
    def getFactory(config, intermediate_serializer = False):
        if config.trie_fname == ':memory:' and not intermediate_serializer:
            return lambda x: MemoryTrieSerializer(
                x, config.trie_serializer_type)
        elif (config.trie_intermediate_storage == ':memory:'
              and intermediate_serializer):
            return lambda x: MemoryTrieSerializer(
                x, config.trie_serializer_type)
        elif config.trie_serializer_type == 'fuzzy':
            return lambda x: TrieFuzzySerializer(x, config)
        elif config.trie_serializer_type == 'reg':
            return lambda x: NodeTrieSerializer(x, config)
        logging.error('No serializer of type %s', config.trie_serializer_type)

class MemoryTrieSerializer(TrieSerializer):
    memory_cache = {}

    def __init__(self, fname, serializer_type):
        super().__init__(fname)
        self.serializer_type = serializer_type

    def serialize(self, trie):
        self.memory_cache[self.fname] = trie

    def deserialize(self):
        trie = self.memory_cache[self.fname]
        return trie.iterate(self.serializer_type)

class BinaryTrieSerializer(TrieSerializer):
    _fmt = '<QQ'
    _fmt_size = struct.calcsize('<QQ')
    str_len_fmt = '<B'
    str_len_fmt_size = struct.calcsize('B')

    def __init__(self, fname, config):
        super().__init__(fname)
        self.max_len = config.max_len
        self.encoding = config.trie_serializer_encoding
        self.toc_chunk_size = config.toc_chunk_size
        self.use_mmap = config.use_mmap

    def do_serialize(self, trie):
        records = 0
        table_of_contents = {}
        toc_start = -1
        with self.open_file('wb') as afile:
            afile.write(struct.pack(self._fmt, 0, 0))
            for item in trie.iterate(self.serializer_type):
                pwd, weight = item
                self.write_record(afile, pwd, weight)
                records += 1
                if records % self.toc_chunk_size == 0:
                    table_of_contents[records] = afile.tell()
            toc_start = afile.tell()
            for key in sorted(table_of_contents.keys()):
                afile.write(struct.pack(self._fmt, key, table_of_contents[key]))
        assert toc_start > 0
        with self.open_file('r+b') as afile:
            logging.info('Wrote %s records to %s', records, self.fname)
            afile.write(struct.pack(self._fmt, records, toc_start))

    def deserialize(self):
        with self.open_file('rb') as afile:
            num_records, toc_start = struct.unpack(
                self._fmt, afile.read(self._fmt_size))
            for _ in range(num_records):
                answer = self.read_record(afile)
                if answer is None:
                    break
                yield answer

    def read_toc(self, afile):
        num_records, toc_start = struct.unpack(
            self._fmt, afile.read(self._fmt_size))
        afile.seek(toc_start)
        toc = {}
        while True:
            chunk = afile.read(self._fmt_size)
            if len(chunk) == 0:
                break
            key, toc_pos = struct.unpack(self._fmt, chunk)
            toc[key] = toc_pos
        return toc, toc_start

    def read_from_pos(self, afile, start_pos, end_pos):
        afile.seek(start_pos)
        while afile.tell() < end_pos:
            item = self.read_record(afile)
            assert item is not None
            yield item

    def random_access(self):
        with self.open_file('rb') as afile_obj:
            if self.use_mmap:
                afile = mmap.mmap(afile_obj.fileno(), 0, prot = mmap.PROT_READ)
            else:
                afile = afile_obj
            toc, toc_start = self.read_toc(afile)
            toc_locations = list(map(lambda k: toc[k], sorted(toc.keys())))
            start_pos = [self._fmt_size] + toc_locations
            end_pos = toc_locations + [toc_start]
            intervals = list(zip(start_pos, end_pos))
            random.shuffle(intervals)
            for interval in intervals:
                start, end = interval
                for item in self.read_from_pos(afile, start, end):
                    yield item

    def read_string(self, afile):
        byte_string = afile.read(self.str_len_fmt_size)
        if len(byte_string) == 0:
            return None
        strlen, = struct.unpack(self.str_len_fmt, byte_string)
        return afile.read(strlen).decode(self.encoding)

    def write_string(self, afile, astring):
        string_bytes = astring.encode(self.encoding)
        try:
            afile.write(struct.pack(self.str_len_fmt, len(string_bytes)))
        except struct.error as e:
            logging.critical('Error when processing string %s', astring)
            raise
        afile.write(string_bytes)

    def write_record(self, ostream, pwd, val):
        self.write_string(ostream, pwd)
        self.write_value(ostream, val)

    def read_record(self, afile):
        astr = self.read_string(afile)
        if astr is None:
            return None
        return (astr, self.read_value(afile))

    def write_value(self, afile, value):
        raise NotImplementedError

    def read_value(self, afile):
        raise NotImplementedError()

class NodeTrieSerializer(BinaryTrieSerializer):
    serializer_type = 'reg'
    def __init__(self, *args):
        super().__init__(*args)
        self.fmt = '<Q'
        self.chunk_size = struct.calcsize(self.fmt)

    def write_value(self, ostream, weight):
        ostream.write(struct.pack(self.fmt, weight))

    def read_value(self, afile):
        return struct.unpack_from(self.fmt, afile.read(self.chunk_size))[0]

class TrieFuzzySerializer(BinaryTrieSerializer):
    serializer_type = 'fuzzy'

    def __init__(self, *args):
        super().__init__(*args)
        self.in_fmt = '<H'
        self.out_fmt = '<1sQ'
        self.in_fmt_bytes = struct.calcsize(self.in_fmt)
        self.out_fmt_bytes = struct.calcsize(self.out_fmt)

    def write_value(self, ostream, output_list):
        ostream.write(struct.pack(self.in_fmt, len(output_list)))
        for item in output_list:
            char, weight = item
            ostream.write(struct.pack(
                self.out_fmt, char.encode(self.encoding), weight))

    def read_value(self, istream):
        num_rec, = struct.unpack_from(
            self.in_fmt, istream.read(self.in_fmt_bytes))
        record = []
        for _ in range(num_rec):
            out_char, out_weight = struct.unpack_from(
                self.out_fmt, istream.read(self.out_fmt_bytes))
            record.append((out_char.decode(self.encoding), out_weight))
        return record

class CharacterTable(object):
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
        x_str_list = map(lambda x: self.pad_to_len(x, maxlen), string_list)
        x_vec = np.zeros((len(string_list), maxlen, len(self.chars)),
                         dtype = np.bool)
        for i, xstr in enumerate(x_str_list):
            self._encode_into(x_vec[i], xstr)
        return x_vec

    def _encode_into(self, X, C):
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        self._encode_into(X, C)
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

    def get_char_index(self, character):
        return self.char_indices[character]

    def translate(self, astring):
        return astring

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
        for key in self.rare_dict:
            self.char_indices[key] = self.char_indices[self.rare_dict[key]]
        translate_table = {}
        for c in chars:
            if c in self.rare_dict:
                translate_table[c] = self.rare_dict[c]
            else:
                translate_table[c] = c
        self.translate_table = ''.maketrans(translate_table)

    def translate(self, astring):
        return astring.translate(self.translate_table)

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

model_type_dict = {
    'JZS1' : recurrent.JZS1,
    'JZS2' : recurrent.JZS2,
    'JZS3' : recurrent.JZS3,
    'GRU' : recurrent.GRU
}

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
    intermediate_fname = ':memory:'
    preprocess_trie_on_disk = False
    preprocess_trie_on_disk_buff_size = 100000
    trie_serializer_encoding = 'utf8'
    trie_serializer_type = 'reg'
    save_always = True
    randomize_training_order = True
    toc_chunk_size = 1000
    model_truncate_gradient = -1
    model_optimizer = 'adam'
    guesser_intermediate_directory = 'guesser_files'
    cleanup_guesser_files = True
    use_mmap = True
    compute_stats = False
    password_test_fname = None
    chunk_size_guesser = 10000

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
        answer.validate()
        return answer

    def validate(self):
        if self.trie_serializer_type == 'fuzzy':
            assert self.simulated_frequency_optimization
            assert self.trie_implementation is not None
        assert self.fork_length < self.min_len
        assert self.max_len <= 255
        if self.guess_serialization_method == 'calculator':
            assert os.path.exists(self.password_test_fname)

    def as_dict(self):
        answer = dict(vars(ModelDefaults).copy())
        answer.update(self.adict)
        return dict([(key, value) for key, value in answer.items() if (
            key[0] != '_' and not hasattr(value, '__call__')
            and not type(value) == staticmethod)])

    def model_type_exec(self):
        try:
            return model_type_dict[self.model_type]
        except KeyError as e:
            logging.error('Cannot find model type %s', self.model_type)

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
        return BasePreprocessor.config_keys[config.trie_implementation](config)

    @staticmethod
    def byFormat(pwd_format, config):
        return BasePreprocessor.format_keys[pwd_format](config)

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
        self.ctable = CharacterTable.fromConfig(config)
        self.ordered_randomly = False

    def preprocess(self, pwd_list):
        return map(lambda x: (self.ctable.translate(x[0]), x[1]), pwd_list)

    def begin(self, pwd_list):
        for item in self.preprocess(pwd_list):
            pwd, weight = item
            self.instances += len(pwd) + 1
            self.trie.increment(pwd + PASSWORD_END, weight)
        self.trie.finish()
        logging.info('Saving preprocessing step...')
        TrieSerializer.fromConfig(self.config).serialize(self.trie)

    def reset(self):
        self.set_iterator()
        if self.config.randomize_training_order and not self.ordered_randomly:
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
            pwd_file if pwd_file is not None else self.config.trie_fname)

    def set_iterator(self):
        self.ordered_randomly = self.config.randomize_training_order
        if self.config.randomize_training_order:
            self.current_generator = self.serializer.random_access()
        else:
            self.current_generator = self.serializer.deserialize()

class IntermediatePreprocessor(TriePreprocessor):
    def begin(self, pwd_file = None):
        logging.info('Loading trie intermediate representation...')
        self.trie = DiskBackedTrie.fromIntermediate(self.config)

class HybridDiskPreprocessor(TriePreprocessor):
    class MemoryCache(object):
        def __init__(self):
            self.cache = collections.defaultdict(list)

        def add_key(self, key, value):
            self.cache[key].append(value)

        def read(self):
            for key in sorted(self.cache.keys()):
                subkeys = self.cache[key]
                for item in subkeys:
                    yield item

    class DiskCache(object):
        def __init__(self, config):
            self.buffer = HybridDiskPreprocessor.MemoryCache()
            self.file_name_mapping = {}
            self.storage_dir = config.trie_intermediate_storage
            self.buff_size = 0
            self.flush_count = 0
            self.chunk_size = config.preprocess_trie_on_disk_buff_size
            if not os.path.exists(self.storage_dir):
                os.mkdir(self.storage_dir)

        def santize(self, key):
            answer = FNAME_PREFIX_PREPROCESSOR + str(
                len(self.file_name_mapping))
            self.file_name_mapping[key] = answer
            return os.path.join(self.storage_dir, answer)

        def unsantize(self, key):
            return os.path.join(self.storage_dir, self.file_name_mapping[key])

        def flush_buff(self):
            logging.info('Flushing %s passwords: num_flushed %s',
                         self.chunk_size, self.flush_count)
            for key in sorted(self.buffer.cache.keys()):
                if key not in self.file_name_mapping:
                    fname = self.santize(key)
                else:
                    fname = self.unsantize(key)
                with open(fname, 'a') as afile:
                    writer = csv.writer(
                        afile, delimiter = '\t', quotechar = None)
                    values = self.buffer.cache[key]
                    for value in values:
                        writer.writerow(value)
            self.buff_size = 0
            self.buffer = HybridDiskPreprocessor.MemoryCache()
            self.flush_count += 1

        def add_key(self, key, value):
            self.buffer.add_key(key, value)
            self.buff_size += 1
            if self.buff_size == self.chunk_size:
                self.flush_buff()

        def read(self):
            if self.buff_size != 0:
                self.flush_buff()
            for key in sorted(self.file_name_mapping.keys()):
                with open(self.unsantize(key), 'r') as istr:
                    for item in csv.reader(
                            istr, delimiter = '\t', quotechar = None):
                        yield (item[0], int(item[1]))

    def preprocess(self, pwd_list):
        if (self.config.trie_intermediate_storage == ':memory:' or
            not self.config.preprocess_trie_on_disk):
            out_pwd_list = HybridDiskPreprocessor.MemoryCache()
        else:
            out_pwd_list = HybridDiskPreprocessor.DiskCache(self.config)
        fork_len = self.config.fork_length
        for item in super().preprocess(pwd_list):
            out_pwd_list.add_key(item[0][:fork_len], item)
        return out_pwd_list.read()

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
            self.config.hidden_size,
            input_shape = (self.config.max_len, len(self.ctable.chars)),
            truncate_gradient = self.config.model_truncate_gradient))
        model.add(RepeatVector(1))
        for _ in range(self.config.layers):
            model.add(self.config.model_type_exec()(
                self.config.hidden_size, return_sequences = True,
                truncate_gradient = self.config.model_truncate_gradient))
        model.add(TimeDistributedDense(len(self.ctable.chars)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer = self.config.model_optimizer)
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
            weight_sum = 0
            for record in records:
                outchar, weight = record
                weight_sum += weight
                y_vec[i, 0, self.ctable.get_char_index(outchar)] = weight
            y_vec[i] /= weight_sum
        return y_vec

class PwdList(object):
    def __init__(self, ifile_name):
        self.ifile_name = ifile_name

    def as_list_iter(self, agen):
        for row in agen:
            yield (row.strip(PASSWORD_END), 1)

    def as_list(self):
        if self.ifile_name[-3:] == '.gz':
            with gzip.open(self.ifile_name, 'rt') as ifile:
                for item in self.as_list_iter(ifile):
                    yield item
        else:
            with open(self.ifile_name, 'r') as ifile:
                for item in self.as_list_iter(ifile):
                    yield item

    @staticmethod
    def getFactory(file_formats, config):
        assert type(file_formats) == list
        if len(file_formats) > 1:
            return lambda flist: ConcatenatingList(config, flist, file_formats)
        assert len(file_formats) > 0
        if file_formats[0] == 'tsv':
            if config.simulated_frequency_optimization:
                return lambda flist: TsvSimulatedList(flist[0])
            else:
                return lambda flist: TsvList(flist[0])
        elif file_formats[0] == 'list':
            return lambda flist: PwdList(flist[0])
        logging.error('Cannot find factory for format of %s', file_formats)

class TsvList(PwdList):
    def as_list_iter(self, agen):
        for row in csv.reader(iter(agen), delimiter = '\t', quotechar = None):
            pwd, freq = row[:2]
            for _ in range(int(float.fromhex(freq))):
                yield (sys.intern(pwd), 1)

class TsvSimulatedList(PwdList):
    def as_list_iter(self, agen):
        for row in csv.reader(iter(agen), delimiter = '\t', quotechar = None):
            yield (row[0], int(float.fromhex(row[1])))

class ConcatenatingList(object):
    def __init__(self, config, file_list, file_formats):
        assert len(file_list) == len(file_formats)
        self.config = config
        self.file_tuples = zip(file_list, file_formats)

    def get_iterable(self, file_tuple):
        file_name, file_format = file_tuple
        input_factory = PwdList.getFactory([file_format], self.config)
        return input_factory([file_name])

    def as_list(self):
        answer = []
        for atuple in self.file_tuples:
            iterable = self.get_iterable(atuple)
            logging.info('Reading from %s', atuple)
            answer.append(iterable.as_list())
        return itertools.chain.from_iterable(answer)

class Filterer(object):
    def __init__(self, config):
        self.filtered_out = 0
        self.total = 0
        self.total_characters = 0
        self.frequencies = collections.defaultdict(int)
        self.config = config
        self.longest_pwd = 0
        self.char_bag = config.char_bag
        self.max_len = config.max_len
        self.min_len = config.min_len

    def pwd_is_valid(self, pwd, quick = False):
        pwd = pwd.strip(PASSWORD_END)
        answer = (all(map(lambda c: c in self.char_bag, pwd)) and
                  len(pwd) <= self.max_len and
                  len(pwd) >= self.min_len)
        if quick:
            return answer
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

    def finish(self, save_stats = True):
        logging.info('Filtered %s of %s passwords',
                     self.filtered_out, self.total)
        char_freqs = {}
        for key in self.frequencies:
            char_freqs[key] = self.frequencies[key] / self.total_characters
        if save_stats:
            self.config.set_intermediate_info(
                'rare_character_bag', self.rare_characters())
            logging.info('Rare characters: %s', self.rare_characters())
            logging.info('Longest pwd is : %s characters long',
                         self.longest_pwd)

    def filter(self, alist):
        return filter(lambda x: self.pwd_is_valid(x[0]), alist)

class GuessSerializer(object):
    def __init__(self, ostream):
        self.ostream = ostream

    def serialize(self, password, prob):
        self.ostream.write('%s\t%s\n' % (
            password.strip(PASSWORD_END), prob))

    def finish(self):
        self.ostream.flush()
        self.ostream.close()

class GuessNumberGenerator(GuessSerializer):
    def __init__(self, ostream, pwd_list):
        super().__init__(ostream)
        self.pwds, self.probs = zip(*sorted(pwd_list, key = lambda x: x[1]))
        self.guess_numbers = [0] * len(self.pwds)
        self.total_guessed = 0

    def serialize(self, _, prob):
        self.total_guessed += 1
        idx = bisect.bisect_left(self.probs, prob) - 1
        if idx >= 0:
            self.guess_numbers[idx] += 1

    def finish(self):
        for i in range(len(self.guess_numbers) - 1, 0, -1):
            self.guess_numbers[i - 1] += self.guess_numbers[i]
        logging.info('Guessed %s passwords', self.total_guessed)
        self.ostream.write('Total count: %s\n' % self.total_guessed)
        writer = csv.writer(self.ostream, delimiter = '\t', quotechar = None)
        for i in range(len(self.pwds), 0, -1):
            idx = i - 1
            writer.writerow([
                self.pwds[idx], self.probs[idx], self.guess_numbers[idx]])
        self.ostream.flush()
        self.ostream.close()

class ProbabilityCalculator(object):
    def __init__(self, guesser):
        self.guesser = guesser
        self.ctable = CharacterTable.fromConfig(guesser.config)
        self.preproc = Preprocessor(guesser.config)

    def probability_stream(self, pwd_list):
        self.preproc.begin(pwd_list)
        x_strings, y_strings, _ = self.preproc.next_chunk()
        while len(x_strings) != 0:
            y_indices = list(map(self.ctable.get_char_index, y_strings))
            probs = self.guesser.conditional_probs_many(x_strings)
            for i in range(len(y_indices)):
                yield x_strings[i], y_strings[i], probs[i][0][y_indices[i]]
            x_strings, y_strings, _ = self.preproc.next_chunk()

    def calc_probabilities(self, pwd_list):
        prev_prob = 1
        for item in self.probability_stream(pwd_list):
            input_string, next_char, output_prob = item
            prev_prob *= output_prob
            if next_char == PASSWORD_END:
                yield (input_string, prev_prob)
                prev_prob = 1

class Guesser(object):
    def __init__(self, model, config, ostream):
        self.model = model
        self.config = config
        self.max_len = config.max_len
        self.lower_probability_threshold = config.lower_probability_threshold
        self.relevel_not_matching_passwords = (
            config.relevel_not_matching_passwords)
        self.generated = 0
        self.ctable = CharacterTable.fromConfig(self.config)
        self.filterer = Filterer(self.config)
        self.output_serializer = self.make_serializer(ostream)
        self.chunk_size_guesser = self.config.chunk_size_guesser

    def make_serializer(self, ostream):
        if self.config.guess_serialization_method == 'human':
            return GuessSerializer(ostream)
        elif self.config.guess_serialization_method == 'calculator':
            logging.info('Reading password calculator test set...')
            filterer = Filterer(self.config)
            pwds = list(filterer.filter(
                PwdList(self.config.password_test_fname).as_list()))
            filterer.finish(save_stats=False)
            logging.info('Calculating test set probabilities')
            return GuessNumberGenerator(
                ostream, ProbabilityCalculator(self).calc_probabilities(pwds))
        logging.error('Unknown serialization method %s',
                      config.guess_serialization_method)

    def cond_prob_from_preds(self, char, preds):
        return preds[self.ctable.get_char_index(char)]

    def relevel_prediction(self, preds, astring):
        if not self.filterer.pwd_is_valid(astring, quick = True):
            preds[self.ctable.get_char_index(PASSWORD_END)] = 0
        elif len(astring) == self.max_len:
            multiply = np.zeros(len(preds))
            pwd_end_idx = self.ctable.get_char_index(PASSWORD_END)
            multiply[pwd_end_idx] = 1
            preds[pwd_end_idx] = 1
            preds = np.multiply(preds, multiply, preds)
        sum_per = sum(preds)
        for i, v in enumerate(preds):
            preds[i] = v / sum_per

    def relevel_prediction_many(self, pred_list, str_list):
        for i in range(len(pred_list)):
            self.relevel_prediction(pred_list[i][0], str_list[i])

    def conditional_probs(self, astring):
        return self.conditional_probs_many([astring])[0][0].copy()

    def _conditional_probs(self, astring, cache):
        answer = cache[astring]
        del cache[astring]
        return answer[0]

    def conditional_probs_many(self, astring_list):
        answer = self.model.predict(self.ctable.encode_many(astring_list),
                                    verbose = 0, self.chunk_size_guesser)
        if self.relevel_not_matching_passwords:
            answer = np.array(answer)
            self.relevel_prediction_many(answer, astring_list)
        return answer

    def next_nodes(self, astring, prob, cache):
        prediction = self._conditional_probs(astring, cache)
        total_preds = np.array(prediction) * prob
        for char in self.ctable.chars:
            chain_pass = astring + char
            chain_prob = total_preds[self.ctable.get_char_index(char)]
            if chain_prob < self.lower_probability_threshold:
                continue
            if char == PASSWORD_END:
                self.output_serializer.serialize(chain_pass, chain_prob)
                self.generated += 1
            elif len(chain_pass) > self.max_len:
                continue
            elif char != PASSWORD_END:
                yield chain_pass, chain_prob

    def next_node_chunker(self, astring, prob, cached_prefixes):
        for next_node in self.next_nodes(astring, prob, cached_prefixes):
            yield next_node

    def super_node_recur(self, node_list):
        prefixes = list(map(lambda x: x[0], node_list))
        if len(prefixes) == 0:
            return
        logging.info('Super node buffer size %s', len(prefixes))
        predictions = self.conditional_probs_many(prefixes)
        cached_prefixes = dict(zip(prefixes, predictions))
        node_batch = []
        for cur_node in node_list:
            astring, prob = cur_node
            for next_node in self.next_node_chunker(
                    astring, prob, cached_prefixes):
                node_batch.append(next_node)
                if len(node_batch) == self.chunk_size_guesser:
                    self.super_node_recur(node_batch)
                    node_batch = []
        if len(node_batch) > 0:
            self.super_node_recur(node_batch)
            node_batch = []

    def recur(self, astring = '', prob = 1):
        self.super_node_recur([(astring, prob)])

    def guess(self, astring = '', prob = 1):
        self.recur(astring, prob)

    def complete_guessing(self, start = '', start_prob = 1):
        logging.info('Enumerating guesses starting at %s, %s...',
                     start, start_prob)
        self.guess(start, start_prob)
        self.output_serializer.finish()
        logging.info('Generated %s guesses', self.generated)
        return self.generated

class GuesserBuilderError(Exception): pass

class GuesserBuilder(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.serializer = None
        self.ostream = None
        self.ofile_path = None
        self.parallel = self.config.parallel_guessing

    def add_model(self, model):
        self.model = model
        return self

    def add_serializer(self, serializer):
        self.serializer = serializer
        return self

    def add_stream(self, ostream):
        self.ostream = ostream
        return self

    def add_file(self, ofname):
        self.add_stream(open(ofname, 'w'))
        self.ofile_path = ofname
        return self

    def add_temp_file(self):
        intm_dir = self.config.guesser_intermediate_directory
        handle, path = tempfile.mkstemp(dir=intm_dir)
        self.ofile_path = path
        return self.add_stream(os.fdopen(handle, 'w'))

    def add_parallel_setting(self, setting):
        self.parallel = setting
        return self

    def build(self):
        if self.parallel:
            model_or_serializer = self.serializer
        else:
            model_or_serializer = self.model
            if self.serializer is not None and self.model is None:
                model_or_serializer = self.serializer.load_model()
        if model_or_serializer is None:
            raise GuesserBuilderError('Cannot build without model')
        if self.ostream is None:
            raise GuesserBuilderError('Cannot build without ostream')
        assert self.config is not None
        class_builder = ParallelGuesser if self.parallel else Guesser
        return class_builder(model_or_serializer, self.config, self.ostream)

def fork_starting_point(args):
    config_dict, serializer_args, node = args
    return ParallelGuesser.fork_entry_point(
        ModelSerializer(*serializer_args).load_model(),
        ModelDefaults(**config_dict), node)

class ParallelGuesser(Guesser):
    def __init__(self, serializer, config, ostream):
        self.tempOstream = tempfile.NamedTemporaryFile(
            mode = 'w', delete = False)
        model = serializer.load_model()
        super().__init__(model, config, self.tempOstream)
        self.fork_points = []
        self.intermediate_files = []
        self.serializer = serializer
        self.real_output = ostream
        self.fork_starter = fork_starting_point

    def super_node_recur(self, node_list):
        continue_nodes = []
        for node in node_list:
            if len(node[0]) == self.config.fork_length:
                self.fork_points.append(node)
            else:
                continue_nodes.append(node)
        super().super_node_recur(continue_nodes)

    def guess(self, astring = '', prob = 1):
        self.recur(astring, prob)
        self.do_forking()

    def prepare_argument_dict(self, node):
        return (self.config.as_dict(), [
                self.serializer.archfile, self.serializer.weightfile],
                node)

    def arg_list(self):
        return list(map(self.prepare_argument_dict, self.fork_points))

    def collect_answer(self, file_name_list):
        for file_name in file_name_list:
            with open(file_name, 'r') as istream:
                logging.info('Collecting guesses from %s', file_name)
                for line in istream:
                    self.real_output.write(line)
            if self.config.cleanup_guesser_files:
                os.remove(file_name)
        if self.config.cleanup_guesser_files:
            try:
                os.rmdir(self.config.guesser_intermediate_directory)
            except OSError as e:
                logging.error('Cannot remove %s because it is not empty. ',
                              self.config.guesser_intermediate_directory)
        self.real_output.flush()

    def do_forking(self):
        arg_list = self.arg_list()
        # Check that the path exists before forking otherwise there are race
        # conditions
        if not os.path.exists(self.config.guesser_intermediate_directory):
            os.mkdir(self.config.guesser_intermediate_directory)
        pool = mp.Pool(min(len(arg_list), mp.cpu_count()),
                       # Important to free resources
                       maxtasksperchild = 1)
        result = pool.map_async(self.fork_starter, arg_list)
        try:
            pool.close()
            pool.join()
            answer = result.get(timeout = 1)
            generated, files = zip(*answer)
            self.generated = sum(generated) + self.generated
            self.collect_answer(files)
        except KeyboardInterrupt as e:
            logging.error('Received keyboard interrupt. Stopping...')
            pool.terminate()

    @staticmethod
    def fork_entry_point(model, config, node):
        builder = (GuesserBuilder(config).add_model(model).add_temp_file()
                   .add_parallel_setting(False))
        guesser = builder.build()
        start_str, start_prob = node
        guesser.complete_guessing(start_str, start_prob)
        return guesser.generated, builder.ofile_path

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
        sys.stderr.write('Uncaught exception!\n %s\n' % (value))
    sys.excepthook = except_hook
    sys.setcheckinterval = 1000
    log_format = '%(asctime)-15s %(levelname)s: %(message)s'
    log_level = log_level_map[args['log_level']]
    if args['log_file']:
        logging.basicConfig(filename = args['log_file'],
                            level = log_level, format = log_format)
    else:
        logging.basicConfig(level = log_level, format = log_format)
    logging.info('Beginning...')
    logging.info('Arguments: %s', json.dumps(args, indent = 4))
    logging.info('Version: %s', get_version_string())

def read_passwords(pwd_file, pwd_format, config):
    input_factory = PwdList.getFactory(pwd_format, config)
    filt = Filterer(config)
    logging.info('Reading training set...')
    for item in filt.filter(input_factory(pwd_file).as_list()):
        pass
    filt.finish()
    logging.info('Done reading passwords...')
    return filt.filter(input_factory(pwd_file).as_list())

def preprocessing(args, config):
    if args['pwd_format'][0] in BasePreprocessor.format_keys:
        assert len(args['pwd_format']) == 1
        logging.info('Formatting preprocessor')
        disk_trie = BasePreprocessor.byFormat(args['pwd_format'][0], config)
        disk_trie.begin(args['pwd_file'][0])
        return disk_trie
    # read_passwords must be called before creating the preprocessor because it
    # initializes statistics needed for some preprocessors
    plist = read_passwords(args['pwd_file'], args['pwd_format'], config)
    preprocessor = BasePreprocessor.fromConfig(config)
    preprocessor.begin(plist)
    if args['pre_processing_only']:
        logging.info('Only performing pre-processing. ')
        if config.compute_stats:
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
    (GuesserBuilder(config).add_serializer(serializer).add_model(trainer.model)
     .add_file(args['enumerate_ofile'])).build().complete_guessing()

def guess(args, config):
    logging.info('Loading model...')
    if args['arch_file'] is None or args['weight_file'] is None:
        logging.error('Architecture file or weight file not found. Quiting...')
        sys.exit(1)
    (GuesserBuilder(config).add_serializer(
        ModelSerializer(args['arch_file'], args['weight_file']))
     .add_file(args['enumerate_ofile'])).build().complete_guessing()

def read_config_args(args):
    config_arg_file = open(args['config_args'], 'r')
    try:
        config_args = json.load(config_arg_file)
    finally:
        config_arg_file.close()
    arg_ret = args.copy()
    arg_ret.update(config_args['args'])
    if 'profile' in config_args['args']:
        logging.warning(('Profile argument must be given at command line. '
                         'Proceeding without profiling. '))
    config = ModelDefaults(config_args['config'])
    config.validate()
    return config, arg_ret

def main(args):
    if args['version']:
        sys.stdout.write(get_version_string() + '\n')
        sys.exit(0)
    if args['config_args']:
        config, args = read_config_args(args)
    else:
        config = ModelDefaults.fromFile(args['config'])
    init_logging(args)
    logging.info('Configuration: %s', json.dumps(config.as_dict(), indent = 4))
    if theano.config.floatX == 'float64':
        logging.warning(('Using float64 instead of float32 for theano will'
                         ' harm performance. Edit ~/.theanorc'))
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
    parser.add_argument('--pwd-file', help=('Input file name. '), nargs = '+')
    parser.add_argument('--arch-file',
                        help = 'Output file for the model architecture. ')
    parser.add_argument('--weight-file',
                        help = 'Output file for the weights of the model. ')
    parser.add_argument('--pwd-format', default = 'list', nargs = '+',
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
    parser.add_argument('--config-args',
                        help = 'File with both configuration and arguments. ')
    return parser

if __name__=='__main__':
    args = vars(make_parser().parse_args())
    main_bundle = lambda: main(args)
    if args['profile'] is not None:
        cProfile.run('main_bundle()', filename = args['profile'])
    else:
        main_bundle()
