# -*- coding: utf-8 -*-
from theano import function, config, shared, sandbox
import theano.tensor as T
from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dense, RepeatVector
from keras.layers import recurrent

import unittest
from unittest.mock import MagicMock, Mock
import tempfile
import shutil
import os.path
import gzip
import io
import json
import numpy as np
import csv
import time
import string
import random

import pwd_guess

class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.archfile = tempfile.NamedTemporaryFile(mode = 'w')
        self.weightfile = tempfile.NamedTemporaryFile(mode = 'r')

    def test_save_load(self):
        model = Sequential()
        model.add(recurrent.JZS1(3, 64))
        model.add(RepeatVector(1))
        for _ in range(2):
            model.add(recurrent.JZS1(64, 64, return_sequences = True))
        model.add(Dense(64, 3))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer = 'adam')
        np_inp = np.array([[[False, False, True],
                            [False, False, True],
                            [True, False, False]]])
        test_output_before_save = model.predict(np_inp, verbose = 0).copy()
        self.archfile.write(model.to_json())
        self.archfile.flush()
        model.save_weights(self.weightfile.name, overwrite = True)

        archfile_reopen = open(self.archfile.name, 'r')
        new_model = model_from_json(archfile_reopen.read())
        archfile_reopen.close()
        new_model.load_weights(self.weightfile.name)
        test_output_after_save = new_model.predict(np_inp, verbose = 0).copy()
        np.testing.assert_array_equal(
            test_output_before_save, test_output_after_save)

class GPUTest(unittest.TestCase):
    def test_using_gpu(self):
        vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
        iters = 1000

        rng = np.random.RandomState(22)
        x = shared(np.asarray(rng.rand(vlen), config.floatX))
        f = function([], T.exp(x))
        f.maker.fgraph.toposort()
        t0 = time.time()
        for i in range(iters):
            r = f()
        t1 = time.time()
        if np.any([isinstance(
                x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
            gpu = False
        else:
            gpu = True
        self.assertTrue(gpu)

class EndToEndTest(unittest.TestCase):
    skewed_dict = ['abab', 'abbbb', 'aaaa', 'aaab']
    probs = [0.1, 0.4, 0.2, 0.3]

    def skewed(self):
        return self.skewed_dict[
            np.random.choice(len(self.skewed_dict), 1, p = self.probs)[0]]

    def make_dist(self, line_count, dist):
        def random_pass():
            if random.random() < 0.3:
                return 'password'
            elif random.random() < 0.5:
                return 'wordpass'
            else:
                return random.sample(string.ascii_lowercase, 11)
        fun = self.skewed if dist == 'skewed' else lambda: 'aaa'
        if dist == 'fake_pwd':
            fun = random_pass
        for _ in range(line_count):
            self.input_file.write('%s\n' % fun())

    def setUp(self):
        self.config_file, self.output_file, self.input_file = (
            tempfile.NamedTemporaryFile(mode = 'w'),
            tempfile.NamedTemporaryFile(mode = 'r'),
            tempfile.NamedTemporaryFile(mode = 'w'))
        self.archfile = tempfile.NamedTemporaryFile(mode = 'w')
        self.weightfile = tempfile.NamedTemporaryFile(mode = 'w')

    def test_skewed(self):
        json.dump({
            "chunk_print_interval" : 100,
            "training_chunk" : 64,
            "layers" : 3,
            "hidden_size" : 128,
            "generations" : 20,
            "min_len" : 3,
            "max_len" : 5,
            "char_bag" : "ab\n"
        }, self.config_file)
        self.make_dist(10000, 'skewed')
        self.config_file.flush()
        self.input_file.flush()
        pwd_guess.main(vars(pwd_guess.make_parser().parse_args([
            '--pwd-file', self.input_file.name,
            '--config', self.config_file.name,
            '--enumerate-ofile', self.output_file.name,
            '--arch-file', self.archfile.name,
            '--weight-file', self.weightfile.name,
            '--log-level', 'error'
        ])))
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(self.output_file, delimiter = '\t')]
        sort_freq = list(
            map(lambda x: x[0],
                sorted(pwd_freq, key = lambda x: x[1], reverse = True)))
        self.assertEqual(['abbbb', 'aaab', 'aaaa', 'abab'], sort_freq[:4])

    def test_constant(self):
        json.dump({
            "chunk_print_interval" : 100,
            "training_chunk" : 64,
            "layers" : 3,
            "hidden_size" : 128,
            "generations" : 20,
            "min_len" : 3,
            "max_len" : 5,
            "char_bag" : "ab\n",
            "parallel_guessing" : False
        }, self.config_file)
        self.make_dist(10000, 'constant')
        self.config_file.flush()
        self.input_file.flush()
        pwd_guess.main(vars(pwd_guess.make_parser().parse_args([
            '--pwd-file', self.input_file.name,
            '--config', self.config_file.name,
            '--enumerate-ofile', self.output_file.name,
            '--arch-file', self.archfile.name,
            '--weight-file', self.weightfile.name,
            '--log-level', 'error'
        ])))
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(self.output_file, delimiter = '\t')]
        sort_freq = list(
            map(lambda x: x[0],
                sorted(pwd_freq, key = lambda x: x[1], reverse = True)))
        self.assertEqual(['aaa'], sort_freq[:1])

    def test_big_char_bag(self):
        json.dump({
            "chunk_print_interval" : 100,
            "training_chunk" : 64,
            "layers" : 3,
            "hidden_size" : 128,
            "generations" : 20,
            "min_len" : 3,
            "max_len" : 15,
            "char_bag" : string.ascii_lowercase + '\n',
            "parallel_guessing" : False
        }, self.config_file)
        self.make_dist(100000, 'fake_pwd')
        self.config_file.flush()
        self.input_file.flush()
        pwd_guess.main(vars(pwd_guess.make_parser().parse_args([
            '--pwd-file', self.input_file.name,
            '--config', self.config_file.name,
            '--enumerate-ofile', self.output_file.name,
            '--arch-file', self.archfile.name,
            '--weight-file', self.weightfile.name,
            '--log-level', 'error'
        ])))
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(self.output_file, delimiter = '\t')]
        sort_freq = list(
            map(lambda x: x[0],
                sorted(pwd_freq, key = lambda x: x[1], reverse = True)))
        self.assertEqual(['wordpass', 'password'], sort_freq[:2])

    def test_constant_small(self):
        json.dump({
            "chunk_print_interval" : 100,
            "training_chunk" : 64,
            "layers" : 2,
            "hidden_size" : 20,
            "generations" : 5,
            "min_len" : 3,
            "max_len" : 5,
            "char_bag" : "ab\n",
            "parallel_guessing" : False
        }, self.config_file)
        self.make_dist(100, 'constant')
        self.config_file.flush()
        self.input_file.flush()
        pwd_guess.main(vars(pwd_guess.make_parser().parse_args([
            '--pwd-file', self.input_file.name,
            '--config', self.config_file.name,
            '--enumerate-ofile', self.output_file.name,
            '--arch-file', self.archfile.name,
            '--weight-file', self.weightfile.name,
            '--log-level', 'error'
        ])))
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(self.output_file, delimiter = '\t')]
        sort_freq = list(
            map(lambda x: x[0],
                sorted(pwd_freq, key = lambda x: x[1], reverse = True)))
        self.assertEqual(['aaa'], sort_freq[:1])

    def test_skewed_simulate_frequency(self):
        json.dump({
            "training_chunk" : 64,
            "layers" : 3,
            "hidden_size" : 128,
            "generations" : 10000,
            "min_len" : 3,
            "max_len" : 5,
            "char_bag" : "ab\n",
            "training_accuracy_threshold": -1,
            "simulated_frequency_optimization" : True,
            "trie_implementation" : 'DB',
            "trie_fname" : ":memory:"
        }, self.config_file)
        self.input_file.write("""aaaa\t20
abbbb\t40
abab\t10
aaab\t30""")
        self.config_file.flush()
        self.input_file.flush()
        pwd_guess.main(vars(pwd_guess.make_parser().parse_args([
            '--pwd-file', self.input_file.name, '--tsv',
            '--config', self.config_file.name,
            '--enumerate-ofile', self.output_file.name,
            '--arch-file', self.archfile.name,
            '--weight-file', self.weightfile.name,
            '--log-level', 'error'
        ])))
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(self.output_file, delimiter = '\t')]
        sort_freq = list(
            map(lambda x: x[0],
                sorted(pwd_freq, key = lambda x: x[1], reverse = True)))
        print(sort_freq)
        self.assertEqual(['abbbb', 'aaab', 'aaaa', 'abab'], sort_freq[:4])

    def test_skewed_simulate_frequency_super(self):
        json.dump({
            "training_chunk" : 64,
            "layers" : 3,
            "hidden_size" : 128,
            "generations" : 10000,
            "min_len" : 3,
            "max_len" : 5,
            "char_bag" : "ab\n",
            "training_accuracy_threshold": -1,
            "simulated_frequency_optimization" : True,
            "trie_implementation" : 'trie',
            "trie_serializer_type" : 'fuzzy',
            "trie_fname" : ":memory:"
        }, self.config_file)
        self.input_file.write("""aaaa\t20
abbbb\t40
baaa\t100
abab\t10
aaab\t30""")
        self.config_file.flush()
        self.input_file.flush()
        pwd_guess.main(vars(pwd_guess.make_parser().parse_args([
            '--pwd-file', self.input_file.name,
            '--pwd-format', 'tsv',
            '--config', self.config_file.name,
            '--enumerate-ofile', self.output_file.name,
            '--arch-file', self.archfile.name,
            '--weight-file', self.weightfile.name,
            '--log-level', 'error'
        ])))
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(self.output_file, delimiter = '\t')]
        sort_freq = list(
            map(lambda x: x[0],
                sorted(pwd_freq, key = lambda x: x[1], reverse = True)))
        print(sort_freq)
        self.assertEqual(['baaa', 'abbbb', 'aaab', 'aaaa', 'abab'],
                         sort_freq[:5])
