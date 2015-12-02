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
import re
import logging
import itertools
import sys

import pwd_guess
import generator

class NodeTrieSerializerTest(unittest.TestCase):
    def setUp(self):
        self.trie = pwd_guess.NodeTrie()
        self.trie.increment('aaa', 1)
        self.trie.increment('aab', 5)
        self.tempfile = tempfile.NamedTemporaryFile()

    def tearDown(self):
        self.tempfile.close()

    def test_save_load(self):
        s = pwd_guess.TrieSerializer.fromConfig(
            pwd_guess.ModelDefaults(trie_fname = self.tempfile.name,
                                    trie_serializer_type = 'reg'))
        self.assertEqual(type(s), pwd_guess.NodeTrieSerializer)
        s.serialize(self.trie)
        s = pwd_guess.TrieSerializer.fromConfig(
            pwd_guess.ModelDefaults(trie_fname = self.tempfile.name,
                                    trie_serializer_type = 'reg'))
        self.assertEqual(type(s), pwd_guess.NodeTrieSerializer)
        self.assertEqual(set([('a', 6), ('aa', 6), ('aaa', 1), ('aab', 5)]),
                         set(s.deserialize()))

    def test_save_load_many(self):
        s = pwd_guess.TrieSerializer.fromConfig(
            pwd_guess.ModelDefaults(trie_fname = self.tempfile.name,
                                    trie_serializer_type = 'reg'))
        self.assertEqual(type(s), pwd_guess.NodeTrieSerializer)
        for astring in itertools.permutations('abcdefg', 6):
            self.trie.increment(''.join(astring))
        items = list(self.trie.iterate('reg'))
        s.serialize(self.trie)
        s = pwd_guess.TrieSerializer.fromConfig(
            pwd_guess.ModelDefaults(trie_fname = self.tempfile.name,
                                    trie_serializer_type = 'reg'))
        self.assertEqual(type(s), pwd_guess.NodeTrieSerializer)
        inorder = list(s.deserialize())
        randomorder = list(s.random_access())
        self.assertEqual(set(items), set(inorder))
        self.assertEqual(set(items), set(randomorder))
        self.assertNotEqual(inorder, randomorder)

class MemoryTrieSerializerTest(unittest.TestCase):
    def test_save_load(self):
        trie = pwd_guess.NodeTrie()
        trie.increment('aaa', 1)
        trie.increment('aab', 5)
        s = pwd_guess.TrieSerializer.fromConfig(pwd_guess.ModelDefaults(
            trie_fname = ':memory:', trie_serializer_type = 'reg'))
        self.assertEqual(type(s), pwd_guess.MemoryTrieSerializer)
        s.serialize(trie)
        s = pwd_guess.TrieSerializer.fromConfig(pwd_guess.ModelDefaults(
            trie_fname = ':memory:', trie_serializer_type = 'reg'))
        self.assertEqual(type(s), pwd_guess.MemoryTrieSerializer)
        self.assertEqual(set([('a', 6), ('aa', 6), ('aaa', 1), ('aab', 5)]),
                         set(s.deserialize()))

class TrieFuzzySerializerTest(unittest.TestCase):
    def setUp(self):
        self.trie = pwd_guess.NodeTrie()
        self.trie.increment('aaa', 1)
        self.trie.increment('aab', 5)
        self.tempfile = tempfile.NamedTemporaryFile()

    def tearDown(self):
        self.tempfile.close()

    def test_save_load(self):
        s = pwd_guess.TrieSerializer.fromConfig(
            pwd_guess.ModelDefaults(trie_fname = self.tempfile.name,
                                    trie_serializer_type = 'fuzzy'))
        self.assertEqual(type(s), pwd_guess.TrieFuzzySerializer)
        s.serialize(self.trie)
        s = pwd_guess.TrieSerializer.fromConfig(
            pwd_guess.ModelDefaults(trie_fname = self.tempfile.name,
                                    trie_serializer_type = 'fuzzy'))
        self.assertEqual(type(s), pwd_guess.TrieFuzzySerializer)
        self.assertEqual([('', [('a', 6)]),
                          ('a', [('a', 6)]),
                          ('aa', [('a', 1),
                                  ('b', 5)])],
                         list(s.deserialize()))

    def test_save_load_many(self):
        s = pwd_guess.TrieSerializer.fromConfig(
            pwd_guess.ModelDefaults(trie_fname = self.tempfile.name,
                                    trie_serializer_type = 'fuzzy'))
        self.assertEqual(type(s), pwd_guess.TrieFuzzySerializer)
        for astring in itertools.permutations('abcdefg', 6):
            self.trie.increment(''.join(astring))
        items = list(self.trie.iterate('fuzzy'))
        s.serialize(self.trie)
        s = pwd_guess.TrieSerializer.fromConfig(
            pwd_guess.ModelDefaults(trie_fname = self.tempfile.name,
                                    trie_serializer_type = 'fuzzy'))
        self.assertEqual(type(s), pwd_guess.TrieFuzzySerializer)
        hashable = lambda x: (x[0], tuple(x[1]))
        inorder = list(s.deserialize())
        randomorder = list(s.random_access())
        self.assertEqual(set(map(hashable, items)), set(map(hashable, inorder)))
        self.assertEqual(set(map(hashable, items)),
                         set(map(hashable, randomorder)))
        self.assertNotEqual(list(map(hashable, inorder)),
                            list(map(hashable, randomorder)))

class NodeTrieTest(unittest.TestCase):
    def setUp(self):
        self.trie = pwd_guess.NodeTrie()

    def test_iterate(self):
        self.trie.increment('aaa', 1)
        self.assertEqual([('a', 1), ('aa', 1), ('aaa', 1)],
                         list(self.trie.iterate('reg')))

    def test_iterate_overlap(self):
        self.trie.increment('aaa', 1)
        self.trie.increment('aab', 5)
        self.assertEqual(set([('a', 6), ('aa', 6), ('aaa', 1), ('aab', 5)]),
                         set(self.trie.iterate('reg')))

    def test_sampled_iterate(self):
        self.trie.increment('aaa', 1)
        self.trie.increment('aab', 5)
        self.assertEqual([('', [('a', 6)]),
                          ('a', [('a', 6)]),
                          ('aa', [('a', 1),
                                  ('b', 5)])],
                         list(self.trie.iterate('fuzzy')))

class DiskBackedTrieTest(unittest.TestCase):
    def test_iterate(self):
        self.trie = pwd_guess.DiskBackedTrie(pwd_guess.ModelDefaults(
            fork_length = 1))
        self.trie.increment('aaa', 1)
        self.assertEqual(pwd_guess.MemoryTrieSerializer,
                         type(self.trie.make_serializer()))
        self.assertEqual(self.trie.current_branch_key, 'a')
        self.assertTrue(self.trie.current_node is not None)
        self.assertEqual(set(self.trie.current_node.iterate('reg')),
                         set([('a', 1), ('aa', 1)]))
        self.assertEqual(set([('a', 1), ('aa', 1), ('aaa', 1)]),
                         set(self.trie.iterate('reg')))

    def test_funky_chars(self):
        self.assertFalse(os.path.exists('trie_storage'))
        try:
            self.trie = pwd_guess.DiskBackedTrie(pwd_guess.ModelDefaults(
                fork_length = 1, trie_intermediate_storage = 'trie_storage'))
            self.trie.increment('aaa', 1)
            self.trie.increment('./a', 1)
            self.assertEqual(pwd_guess.NodeTrieSerializer,
                             type(self.trie.make_serializer()))
            self.assertTrue(self.trie.current_node is not None)
            self.assertEqual(set([('a', 1), ('aa', 1), ('aaa', 1),
                                  ('.', 1), ('./', 1), ('./a', 1)]),
                             set(self.trie.iterate('reg')))
        finally:
            shutil.rmtree('trie_storage')

    def test_iterate_overlap(self):
        self.trie = pwd_guess.DiskBackedTrie(pwd_guess.ModelDefaults())
        self.trie.increment('aaa', 1)
        self.trie.increment('aab', 5)
        self.trie.increment('cab', 2)
        self.assertEqual(set([('a', 6), ('c', 2), ('aa', 6), ('aaa', 1),
                              ('aab', 5), ('ca', 2), ('cab', 2)]),
                         set(self.trie.iterate('reg')))

    def test_sampled_iterate(self):
        self.trie = pwd_guess.DiskBackedTrie(pwd_guess.ModelDefaults(
            trie_serializer_type = 'fuzzy'))
        self.trie.increment('aaa', 1)
        self.trie.increment('aab', 5)
        self.trie.increment('cab', 2)
        hashable = lambda item: (item[0], tuple(item[1]))
        self.assertEqual(set(map(hashable, [('', [('a', 6), ('c', 2)]),
                                            ('c', [('a', 2)]),
                                            ('ca', [('b', 2)]),
                                            ('a', [('a', 6)]),
                                            ('aa', [('a', 1), ('b', 5)])])),
                         set(map(hashable, self.trie.iterate('fuzzy'))))

class CharacterTableTest(unittest.TestCase):
    def test_table(self):
        ctable = pwd_guess.CharacterTable('ab', 2)
        np.testing.assert_array_equal(ctable.encode('aa'),
                                      np.array([[True, False],
                                                [True, False]]))
        np.testing.assert_array_equal(ctable.encode('ba'),
                                      np.array([[False, True],
                                                [True, False]]))
        self.assertEqual(ctable.decode(np.array([[True, False],
                                                 [True, False]])), 'aa')
        self.assertEqual(ctable.decode(np.array([[False, True],
                                                 [True, False]])), 'ba')
        self.assertEqual(ctable.get_char_index('a'), 0)
        self.assertEqual(ctable.get_char_index('b'), 1)
        np.testing.assert_array_equal(ctable.encode_many(['aa', 'ba']),
                                      np.array([[[True, False],
                                                 [True, False]],
                                                [[False, True],
                                                 [True, False]]]))
        self.assertEqual(ctable.translate('aba'), 'aba')
        self.assertEqual(ctable.translate('aba'), 'aba')

class OptimizingTableTest(unittest.TestCase):
    def test_table(self):
        ctable = pwd_guess.OptimizingCharacterTable('abcd', 2, 'ab', False)
        self.assertEqual(ctable.rare_character_preimage, {
            'a' : ['a', 'b']
        })
        np.testing.assert_array_equal(ctable.encode('cc'),
                                      np.array([[False, True, False],
                                                [False, True, False]]))
        np.testing.assert_array_equal(ctable.encode('dc'),
                                      np.array([[False, False, True],
                                                [False, True, False]]))
        np.testing.assert_array_equal(ctable.encode('aa'),
                                      np.array([[True, False, False],
                                                [True, False, False]]))
        self.assertEqual(ctable.decode(np.array([[True, False, False],
                                                 [True, False, False]])), 'aa')
        self.assertEqual(ctable.decode(np.array([[False, False, True],
                                                 [False, True, False]])), 'dc')
        self.assertEqual(ctable.get_char_index('a'), 0)
        self.assertEqual(ctable.get_char_index('b'), 0)
        self.assertEqual(ctable.get_char_index('c'), 1)
        self.assertEqual(ctable.get_char_index('d'), 2)

        self.assertEqual(ctable.translate('cdb'), 'cda')
        self.assertEqual(ctable.translate('cda'), 'cda')

    def test_table_upper(self):
        ctable = pwd_guess.OptimizingCharacterTable('abcdABCD:', 2, ':', True)
        self.assertEqual(ctable.rare_character_preimage, {
            ':' : [':'],
            'a' : ['A', 'a'],
            'b' : ['B', 'b'],
            'c' : ['C', 'c'],
            'd' : ['D', 'd']
        })
        np.testing.assert_array_equal(ctable.encode('cc'), np.array(
            [[False, False, False, True, False],
             [False, False, False, True, False]]))
        np.testing.assert_array_equal(ctable.encode('dc'), np.array(
            [[False, False, False, False, True],
             [False, False, False, True, False]]))
        np.testing.assert_array_equal(ctable.encode('dC'), np.array(
            [[False, False, False, False, True],
             [False, False, False, True, False]]))
        np.testing.assert_array_equal(ctable.encode('AA'), np.array(
            [[False, True, False, False, False],
             [False, True, False, False, False]]))
        self.assertEqual(ctable.decode(np.array(
            [[True, False, False, False, False],
             [True, False, False, False, False]])), '::')
        self.assertEqual(ctable.decode(np.array(
            [[False, False, False, False, True],
             [False, False, False, True, False]])), 'dc')
        self.assertEqual(ctable.get_char_index('a'), 1)
        self.assertEqual(ctable.get_char_index('b'), 2)
        self.assertEqual(ctable.get_char_index('c'), 3)
        self.assertEqual(ctable.get_char_index('d'), 4)
        self.assertEqual(ctable.get_char_index(':'), 0)
        self.assertEqual(ctable.get_char_index('B'), 2)
        self.assertEqual(ctable.get_char_index('D'), 4)

    def test_table_upper_and_rare(self):
        ctable = pwd_guess.OptimizingCharacterTable('abcdABCD:', 2, ':A', True)
        self.assertEqual(ctable.rare_characters, ':')
        self.assertEqual(ctable.decode(np.array(
            [[True, False, False, False, False],
             [True, False, False, False, False]])), '::')

class HybridPreprocessorTest(unittest.TestCase):
    def test_begin(self):
        pre = pwd_guess.HybridDiskPreprocessor(pwd_guess.ModelDefaults(
            min_len = 3, intermediate_fname = ':memory:',
            trie_implementation = 'disk'))
        pre.begin([('aaa', 1), ('caa', 2), ('aab', 5)])
        pre.reset()
        prefix, suffix, weight = pre.next_chunk()
        exp_prefix = ['', 'a', 'aa', 'aa', '', 'c', 'ca', 'aaa', 'aab', 'caa']
        exp_suffix = ['a', 'a', 'a', 'b', 'c', 'a', 'a', '\n', '\n', '\n']
        exp_weight = [6, 6, 1, 5, 2, 2, 2, 1, 5, 2]
        self.assertEqual(
            set(zip(exp_prefix, exp_suffix, exp_weight)),
            set(zip(prefix, suffix, weight)))

    def test_preprocess(self):
        pre = pwd_guess.HybridDiskPreprocessor(pwd_guess.ModelDefaults(
            min_len = 3, trie_serializer_type = 'fuzzy',
            trie_implementation = 'disk', intermediate_fname = ':memory:'))
        self.assertEqual([('aaa', 1), ('aab', 5), ('caa', 2)], list(
            pre.preprocess([('aaa', 1), ('caa', 2), ('aab', 5)])))

    def test_preprocess_disk(self):
        tf = tempfile.mkdtemp()
        try:
            pre = pwd_guess.HybridDiskPreprocessor(pwd_guess.ModelDefaults(
                min_len = 3, trie_serializer_type = 'fuzzy',
                trie_implementation = 'disk', intermediate_fname = ':memory:',
                trie_intermediate_storage = tf,
                preprocess_trie_on_disk = True,
                preprocess_trie_on_disk_buff_size = 2))
            self.assertEqual([('aaa', 1), ('aab', 5), ('caa', 2)], list(
                pre.preprocess([('aaa', 1), ('caa', 2), ('aab', 5)])))
        finally:
            shutil.rmtree(tf)

class DiskPreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.tempfile = tempfile.NamedTemporaryFile()
        self.config = pwd_guess.ModelDefaults(trie_fname = self.tempfile.name)

    def tearDown(self):
        self.tempfile.close()

    def test_train(self):
        s = pwd_guess.TrieSerializer.fromConfig(self.config)
        trie = pwd_guess.NodeTrie()
        trie.increment('aaa', 1)
        trie.increment('aab', 5)
        s.serialize(trie)

        pre = pwd_guess.DiskPreprocessor(self.config)
        pre.begin()
        pre.reset()
        prefix, suffix, weight = pre.next_chunk()
        exp_prefix = ['', 'a', 'aa', 'aa']
        exp_suffix = ['a', 'a', 'a', 'b']
        exp_weight = [6, 6, 1, 5]
        self.assertEqual(
            set(zip(exp_prefix, exp_suffix, exp_weight)),
            set(zip(prefix, suffix, weight)))

class PreprocessorTest(unittest.TestCase):
    def test_train_set(self):
        p = pwd_guess.Preprocessor(pwd_guess.ModelDefaults(max_len = 40))
        p.begin([('pass', 1)])
        prefix, suffix, weight = p.next_chunk()
        self.assertEqual(set(zip(['', 'p', 'pa', 'pas', 'pass'],
                                 ['p', 'a', 's', 's', '\n'])),
                         set(zip(list(prefix), list(suffix))))
        self.assertTrue(all(map(lambda x: x == 1, weight)))

    def test_training_set_small(self):
        t = pwd_guess.Preprocessor(
            pwd_guess.ModelDefaults(max_len = 3, min_len = 3))
        t.begin([('aaa', 1)])
        prefix, suffix, _ = t.next_chunk()
        self.assertEqual(set(zip(['', 'a', 'aa', 'aaa'],
                                 ['a', 'a', 'a', '\n'])),
                         set(zip(list(prefix), list(suffix))))

    def train_construct_dict(self):
        t = pwd_guess.Preprocessor({'pass' : 2}, pwd_guess.ModelDefaults(
            simulated_frequency_optimization = True))
        prefix, suffix, weight = p.next_chunk()
        self.assertEqual((['', 'p', 'pa', 'pas', 'pass'],
                          ['p', 'a', 's', 's', '\n'],
                          [2, 2, 2, 2, 2]),
                         (prefix, suffix, weight))
        t = pwd_guess.Preprocessor({'pass' : 2}, pwd_guess.ModelDefaults(
            simulated_frequency_optimization = False))
        prefix, suffix, weight = p.next_chunk()
        self.assertEqual((['', 'p', 'pa', 'pas', 'pass'],
                          ['p', 'a', 's', 's', '\n'],
                          [1, 1, 1, 1, 1]),
                         (prefix, suffix, weight))

class TriePreprocessorTest(unittest.TestCase):
    def test_train_set(self):
        config = pwd_guess.ModelDefaults(
            max_len = 40, trie_implementation = 'trie')
        trie_p = pwd_guess.TriePreprocessor(config)
        trie_p.begin([('pass', 1)])
        trie_p.reset()
        prefix, suffix, weight = trie_p.next_chunk()
        self.assertEqual(set(zip(['', 'p', 'pa', 'pas', 'pass'],
                                 ['p', 'a', 's', 's', '\n'],
                                 [1, 1, 1 ,1 , 1])),
                         set(zip(prefix, suffix, weight)))

    def test_training_set_small(self):
        config = pwd_guess.ModelDefaults(
            max_len = 3, min_len = 3, trie_implementation = 'trie')
        trie_p = pwd_guess.TriePreprocessor(config)
        trie_p.begin([('aaa', 1)])
        trie_p.reset()
        prefix, suffix, weight = trie_p.next_chunk()
        self.assertEqual(set(zip(['', 'a', 'aa', 'aaa'], ['a', 'a', 'a', '\n'],
                                 [1, 1, 1, 1])),
                         set(zip(prefix, suffix, weight)))

    def test_train_construct_dict(self):
        config = pwd_guess.ModelDefaults(
            simulated_frequency_optimization = True,
            trie_implementation = 'trie')
        p = pwd_guess.TriePreprocessor(config)
        p.begin([('pass', 2)])
        p.reset()
        prefix, suffix, weight = p.next_chunk()
        self.assertEqual(set(zip(['', 'p', 'pa', 'pas', 'pass'],
                                 ['p', 'a', 's', 's', '\n'],
                                 [2, 2, 2, 2, 2])),
                         set(zip(prefix, suffix, weight)))

    def test_train_trie_dict_trie(self):
        config = pwd_guess.ModelDefaults(
            simulated_frequency_optimization = True,
            trie_implementation = 'trie',
            chunk_print_interval = 1)
        p = pwd_guess.TriePreprocessor(config)
        p.begin([('pass', 2), ('pasw', 3)])
        p.reset()
        prefix, suffix, weight = p.next_chunk()
        expected_chunks = ['', 'p', 'pa', 'pas', 'pasw', 'pas', 'pass']
        expected_out = ['p', 'a', 's', 'w', '\n', 's', '\n']
        expected_weight = [5, 5, 5, 3, 3, 2, 2]
        self.assertEqual(
            set(zip(expected_chunks, expected_out, expected_weight)),
            set(zip(prefix, suffix, weight)))

    def test_train_trie_rare_character(self):
        self.assertFalse(os.path.exists('test.sqlite'))
        config = pwd_guess.ModelDefaults(
            simulated_frequency_optimization = True,
            uppercase_character_optimization = True,
            trie_implementation = 'trie',
            chunk_print_interval = 1,
            intermediate_fname = 'test.sqlite')
        try:
            config.set_intermediate_info('rare_character_bag', '!@#$')
            p = pwd_guess.TriePreprocessor(config)
            p.begin([('pass', 2), ('pasw', 3)])
            p.reset()
            prefix, suffix, weight = p.next_chunk()
            expected_chunks = ['', 'p', 'pa', 'pas', 'pasw', 'pas', 'pass']
            expected_out = ['p', 'a', 's', 'w', '\n', 's', '\n']
            expected_weight = [5, 5, 5, 3, 3, 2, 2]
            self.assertEqual(
                set(zip(expected_chunks, expected_out, expected_weight)),
                set(zip(prefix, suffix, weight)))
        finally:
            os.remove('test.sqlite')

class SuperTrieTrainerTest(unittest.TestCase):
    def test_y_data(self):
        a = pwd_guess.FuzzyTrieTrainer([], pwd_guess.ModelDefaults(
            max_len = 5, char_bag = 'abc\n'))
        answer = a.prepare_y_data([[('a', 1), ('b', 5), ('c', 2)]])
        expected = np.zeros((1, 1, 4))
        expected[0, 0, 0] = 0
        expected[0, 0, 1] = 1 / 8
        expected[0, 0, 2] = 5 / 8
        expected[0, 0, 3] = 2 / 8
        np.testing.assert_array_equal(answer, expected)

class TrainerTest(unittest.TestCase):
    def test_accuracy(self):
        config = pwd_guess.ModelDefaults(max_len = 5)
        pre = pwd_guess.Preprocessor(config)
        pre.begin([('pass', 1)])
        t = pwd_guess.Trainer(pre, config)
        mock_model = Mock()
        mock_model.train_on_batch = MagicMock(return_value = (0.5, 0.5))
        mock_model.test_on_batch = MagicMock(return_value = (0.5, 0.5))
        t.model = mock_model
        self.assertEqual(0.5, t.train_model_generation())

    def test_train_model(self):
        config = pwd_guess.ModelDefaults(max_len = 5, generations = 20)
        pre = pwd_guess.Preprocessor(config)
        pre.begin([('pass', 1)])
        t = pwd_guess.Trainer(pre, config)
        mock_model = Mock()
        mock_model.train_on_batch = MagicMock(return_value = (0.5, 0.5))
        mock_model.test_on_batch = MagicMock(return_value = (0.5, 0.5))
        t.model = mock_model
        t.train_model(pwd_guess.ModelSerializer())
        self.assertEqual(t.generation, 2)

    def test_char_table_no_error(self):
        t = pwd_guess.Trainer(None)
        self.assertNotEqual(None, t.ctable)
        t.ctable.encode('1234' + ('\n' * 36), 40)

    def test_output_as_np(self):
        pre = pwd_guess.Preprocessor()
        pre.begin([('pass', 1)])
        t = pwd_guess.Trainer(pre)
        t.next_train_set_as_np()

    def test_build_model(self):
        t = pwd_guess.Trainer(['pass'], pwd_guess.ModelDefaults(
            hidden_size = 12, layers = 1))
        t.build_model()
        self.assertNotEqual(None, t.model)

    def test_train_set_np_two(self):
        config = pwd_guess.ModelDefaults()
        pre = pwd_guess.Preprocessor(config)
        pre.begin([('pass', 1), ('word', 1)])
        t = pwd_guess.Trainer(pre, config)
        t.next_train_set_as_np()

    def test_test_set(self):
        config = pwd_guess.ModelDefaults(train_test_ratio = 10)
        t = pwd_guess.Trainer(pwd_guess.Preprocessor(config), config)
        a = np.zeros((10, 1, 1), dtype = np.bool)
        b = np.zeros((10, 1, 1), dtype = np.bool)
        w = np.zeros((10, 1))
        x_t, x_v, y_t, y_v, w_t, w_v = t.test_set(a, b, w)
        self.assertEqual(9, len(x_t))
        self.assertEqual(1, len(x_v))
        self.assertEqual(9, len(y_t))
        self.assertEqual(1, len(y_v))
        self.assertEqual(9, len(w_t))
        self.assertEqual(1, len(w_v))

    def test_test_set_small(self):
        t = pwd_guess.Trainer(pwd_guess.Preprocessor([]),
                              pwd_guess.ModelDefaults(train_test_ratio = 10))
        a = np.zeros((5, 1, 1), dtype = np.bool)
        b = np.zeros((5, 1, 1), dtype = np.bool)
        w = np.zeros((5, 1), dtype = np.bool)
        x_t, x_v, y_t, y_v, w_t, w_v = t.test_set(a, b, w)
        self.assertEqual(4, len(x_t))
        self.assertEqual(1, len(x_v))
        self.assertEqual(4, len(y_t))
        self.assertEqual(1, len(y_v))
        self.assertEqual(4, len(w_t))
        self.assertEqual(1, len(w_v))

    def test_get_factory(self):
        self.assertEqual(pwd_guess.FuzzyTrieTrainer,
                         pwd_guess.Trainer.getFactory(pwd_guess.ModelDefaults(
                             trie_serializer_type = 'fuzzy')))
        self.assertEqual(pwd_guess.Trainer, pwd_guess.Trainer.getFactory(
            pwd_guess.ModelDefaults(trie_serializer_type = 'reg')))

class ModelDefaultsTest(unittest.TestCase):
    def test_get_default(self):
        m = pwd_guess.ModelDefaults()
        self.assertEqual(pwd_guess.ModelDefaults.hidden_size, m.hidden_size)

    def test_get_set(self):
        m = pwd_guess.ModelDefaults(hidden_size = 8)
        self.assertEqual(8, m.hidden_size)
        m = pwd_guess.ModelDefaults()
        self.assertEqual(pwd_guess.ModelDefaults.hidden_size, m.hidden_size)

    def test_get_set_dict(self):
        m = pwd_guess.ModelDefaults({'hidden_size' : 8})
        self.assertEqual(8, m.hidden_size)

    def test_as_dict(self):
        m = pwd_guess.ModelDefaults()
        self.assertTrue(m.as_dict()['hidden_size'],
                        pwd_guess.ModelDefaults.hidden_size)

    def test_serialize_dict(self):
        m = pwd_guess.ModelDefaults()
        self.assertTrue(json.dumps(m.as_dict()) is not None)

    def test_model_type(self):
        m = pwd_guess.ModelDefaults()
        self.assertTrue(hasattr(m.model_type_exec(), '__call__'))

    def test_set(self):
        m = pwd_guess.ModelDefaults()
        m.test = 1
        m.hidden_size = 444
        self.assertEqual(1, m.test)
        other = pwd_guess.ModelDefaults()
        self.assertFalse(hasattr(other, 'test'))
        self.assertNotEqual(other.hidden_size, 444)

    def test_intermediate_files(self):
        with tempfile.NamedTemporaryFile() as intermediate_file:
            m = pwd_guess.ModelDefaults(
                intermediate_fname = intermediate_file.name)
            m.set_intermediate_info('test', 8)
            self.assertEqual(m.get_intermediate_info('test'), 8)
            m = pwd_guess.ModelDefaults(
                intermediate_fname = intermediate_file.name)
            self.assertEqual(m.get_intermediate_info('test'), 8)

class PwdListTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.fcontent = 'pass \nword\n'

    def make_file(self, fname, opener):
        self.fname = os.path.join(self.tempdir, fname)
        t = opener(self.fname, 'wt')
        t.write(self.fcontent)
        t.close()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_gzip_file(self):
        self.make_file('test.txt.gz', gzip.open)
        pwd = pwd_guess.PwdList(self.fname)
        self.assertEqual([('pass ', 1), ('word', 1)], list(pwd.as_list()))

    def test_as_list(self):
        self.make_file('test.txt', open)
        pwd = pwd_guess.PwdList(self.fname)
        self.assertEqual([('pass ', 1), ('word', 1)], list(pwd.as_list()))

    def test_tsv(self):
        self.fcontent = 'pass \t1\tR\nword\t1\tR\n'
        self.make_file('test.tsv', open)
        pwd = pwd_guess.TsvList(self.fname)
        self.assertEqual([('pass ', 1), ('word', 1)], list(pwd.as_list()))

    def test_tsv_multiplier(self):
        self.fcontent = 'pass \t2\tR\nword\t1\tR\n'
        self.make_file('test.tsv', open)
        pwd = pwd_guess.TsvList(self.fname)
        self.assertEqual([('pass ', 1), ('pass ', 1), ('word', 1)],
                         list(pwd.as_list()))

    def test_tsv_quote_char(self):
        self.fcontent = 'pass"\t1\tR\nword\t1\tR\n'
        self.make_file('test.tsv', open)
        pwd = pwd_guess.TsvList(self.fname)
        self.assertEqual([('pass"', 1), ('word', 1)], list(pwd.as_list()))

    def test_tsv_simulated(self):
        self.fcontent = 'pass"\t1\tR\nword\t2\tR\n'
        self.make_file('test.tsv', open)
        pwd = pwd_guess.TsvSimulatedList(self.fname)
        self.assertEqual([('pass"', 1), ('word', 2)], list(pwd.as_list()))

    def test_factory(self):
        self.assertEqual(
            type(pwd_guess.PwdList.getFactory(['tsv'], pwd_guess.ModelDefaults(
                simulated_frequency_optimization = True))(['stuff'])),
            pwd_guess.TsvSimulatedList)
        self.assertEqual(
            type(pwd_guess.PwdList.getFactory(['tsv'], pwd_guess.ModelDefaults(
                simulated_frequency_optimization = False))(['stuff'])),
            pwd_guess.TsvList)
        self.assertEqual(
            type(pwd_guess.PwdList.getFactory(['list'], pwd_guess.ModelDefaults(
                simulated_frequency_optimization = False))(['stuff'])),
            pwd_guess.PwdList)
        self.assertEqual(
            type(pwd_guess.PwdList.getFactory(
                ['list', 'list'], pwd_guess.ModelDefaults(
                    simulated_frequency_optimization = False))([
                        'stuff', 'stuff'])),
            pwd_guess.ConcatenatingList)

    def test_concat(self):
        fact = pwd_guess.PwdList.getFactory(
            ['tsv', 'list'], pwd_guess.ModelDefaults(
                simulated_frequency_optimization = False))
        concat_list = fact([os.path.join(self.tempdir, 'test.tsv'),
                            os.path.join(self.tempdir, 'test.txt')])
        self.assertEqual(type(concat_list), pwd_guess.ConcatenatingList)
        self.fcontent = 'pass \t1\tR\nword\t1\tR\n'
        self.make_file('test.tsv', open)
        self.fcontent = 'pass\nword\n'
        self.make_file('test.txt', open)
        self.assertEqual(list(concat_list.as_list()), [
            ('pass ', 1), ('word', 1), ('pass', 1), ('word', 1)])

    def test_concat_second_time(self):
        with tempfile.NamedTemporaryFile() as tf:
            fnames = [os.path.join(self.tempdir, 'test.tsv'),
                      os.path.join(self.tempdir, 'test.txt')]
            config = pwd_guess.ModelDefaults(
                intermediate_fname = tf.name,
                simulated_frequency_optimization = False)
            config.set_intermediate_info(
                pwd_guess.ConcatenatingList.CONFIG_IM_KEY, {
                    fnames[0] : 1,
                    fnames[1] : .2
                })
            fact = pwd_guess.PwdList.getFactory(['tsv', 'list'], config)
            concat_list = fact(fnames)
            self.assertEqual(type(concat_list), pwd_guess.ConcatenatingList)
            self.fcontent = 'pass \t1\tR\nword\t1\tR\n'
            self.make_file('test.tsv', open)
            self.fcontent = 'pass\nword9\n'
            self.make_file('test.txt', open)
            self.assertEqual(list(concat_list.as_list()), [
                ('pass ', 1), ('word', 1), ('pass', .2), ('word9', .2)])

    def test_concat_twice_same_weights(self):
        with tempfile.NamedTemporaryFile() as tf:
            config = pwd_guess.ModelDefaults(
                intermediate_fname = tf.name,
                simulated_frequency_optimization = False)
            fnames = [os.path.join(self.tempdir, 'test.tsv'),
                      os.path.join(self.tempdir, 'test.txt')]
            fact = pwd_guess.PwdList.getFactory(['tsv', 'list'], config)
            concat_list = fact(fnames)
            self.assertEqual(type(concat_list), pwd_guess.ConcatenatingList)
            self.fcontent = 'pass \t1\tR\nword\t1\tR\n'
            self.make_file('test.tsv', open)
            self.fcontent = 'pass\nword9\n'
            self.make_file('test.txt', open)
            self.assertEqual(list(concat_list.as_list()), [
                ('pass ', 1), ('word', 1), ('pass', 1), ('word9', 1)])
            concat_list.finish()
            self.assertEqual(list(fact(fnames).as_list()), [
                ('pass ', 1), ('word', 1), ('pass', 1), ('word9', 1)])

    def test_concat_twice_diff_weights(self):
        with tempfile.NamedTemporaryFile() as tf:
            fnames = [os.path.join(self.tempdir, 'test.tsv'),
                      os.path.join(self.tempdir, 'test.txt')]
            config = pwd_guess.ModelDefaults(
                intermediate_fname = tf.name,
                pwd_list_weights = {
                    fnames[0] : 1,
                    fnames[1] : .2
                },
                simulated_frequency_optimization = False)
            fact = pwd_guess.PwdList.getFactory(['tsv', 'list'], config)
            concat_list = fact(fnames)
            self.assertEqual(type(concat_list), pwd_guess.ConcatenatingList)
            self.fcontent = 'pass \t1\tR\nword\t1\tR\n'
            self.make_file('test.tsv', open)
            self.fcontent = 'pass\nword9\n'
            self.make_file('test.txt', open)
            self.assertEqual(list(concat_list.as_list()), [
                ('pass ', 1), ('word', 1), ('pass', 1), ('word9', 1)])
            concat_list.finish()
            self.assertEqual(list(fact(fnames).as_list()), [
                ('pass ', 1), ('word', 1), ('pass', .2), ('word9', .2)])

    def test_concat_three_diff_weights(self):
        with tempfile.NamedTemporaryFile() as tf:
            fnames = [os.path.join(self.tempdir, 'test.tsv'),
                      os.path.join(self.tempdir, 'test.txt'),
                      os.path.join(self.tempdir, 'test1.txt')]
            config = pwd_guess.ModelDefaults(
                intermediate_fname = tf.name,
                pwd_list_weights = {
                    fnames[0] : 1,
                    fnames[1] : 1,
                    fnames[2] : 1
                },
                simulated_frequency_optimization = False)
            fact = pwd_guess.PwdList.getFactory(['tsv', 'list', 'list'], config)
            concat_list = fact(fnames)
            self.assertEqual(type(concat_list), pwd_guess.ConcatenatingList)
            self.fcontent = 'pass \t1\tR\nword\t1\tR\n'
            self.make_file('test.tsv', open)
            self.fcontent = 'pass\nword9\n'
            self.make_file('test.txt', open)
            self.fcontent = 'pass\nword9\n'
            self.make_file('test1.txt', open)
            self.assertEqual(list(concat_list.as_list()), [
                ('pass ', 1), ('word', 1), ('pass', 1), ('word9', 1),
                ('pass', 1), ('word9', 1)])
            concat_list.finish()
            self.assertEqual(list(fact(fnames).as_list()), [
                ('pass ', 1), ('word', 1), ('pass', 1), ('word9', 1),
                ('pass', 1), ('word9', 1)])

    def test_concat_twice_lopside(self):
        with tempfile.NamedTemporaryFile() as tf:
            fnames = [os.path.join(self.tempdir, 'test.tsv'),
                      os.path.join(self.tempdir, 'test.txt')]
            config = pwd_guess.ModelDefaults(
                intermediate_fname = tf.name,
                pwd_list_weights = {
                    fnames[0] : 1,
                    fnames[1] : .2
                },
                simulated_frequency_optimization = False)
            fact = pwd_guess.PwdList.getFactory(['tsv', 'list'], config)
            concat_list = fact(fnames)
            self.assertEqual(type(concat_list), pwd_guess.ConcatenatingList)
            self.fcontent = 'pass \t1\tR\nword\t1\tR\n'
            self.make_file('test.tsv', open)
            self.fcontent = 'pass\nword9\nppppp\n'
            self.make_file('test.txt', open)
            self.assertEqual(list(concat_list.as_list()), [
                ('pass ', 1), ('word', 1), ('pass', 1),
                ('word9', 1), ('ppppp', 1)])
            concat_list.finish()
            self.assertEqual(list(fact(fnames).as_list()), [
                ('pass ', 1.25), ('word', 1.25), ('pass', 0.16666666666666669),
                ('word9', 0.16666666666666669), ('ppppp', 0.16666666666666669)])

class FiltererTest(unittest.TestCase):
    def test_pwd_is_valid(self):
        f = pwd_guess.Filterer(pwd_guess.ModelDefaults())
        # Normal characters are good
        self.assertTrue(f.pwd_is_valid('pass'))
        self.assertTrue(f.pwd_is_valid('passWord'))
        self.assertTrue(f.pwd_is_valid('pass$@!@#Word'))
        self.assertTrue(f.pwd_is_valid('1234'))
        self.assertTrue(f.pwd_is_valid('u' * 40))

        # No funky characters
        self.assertFalse(f.pwd_is_valid('£jfiei'))
        # No long passwords
        self.assertFalse(f.pwd_is_valid('u' * 1000))
        # No short passwords
        self.assertFalse(f.pwd_is_valid('u'))
        self.assertEqual(8, f.total)
        self.assertEqual(3, f.filtered_out)

    def test_filter(self):
        f = pwd_guess.Filterer(pwd_guess.ModelDefaults())
        self.assertEqual([('pass', 1)],
                         list(f.filter([('asdf£jfj', 1), ('pass', 1)])))

    def test_filter_uniquify(self):
        f = pwd_guess.Filterer(pwd_guess.ModelDefaults(), True)
        self.assertTrue(f.uniquify)
        self.assertEqual([('pass', 1)], list(f.filter(
            [('asdf£jfj', 1), ('pass', 1), ('pass', 1)])))
        f = pwd_guess.Filterer(pwd_guess.ModelDefaults(), False)
        self.assertEqual([('pass', 1), ('pass', 1)], list(f.filter(
            [('asdf£jfj', 1), ('pass', 1), ('pass', 1)])))

    def test_filter_twice(self):
        f = pwd_guess.Filterer(pwd_guess.ModelDefaults())
        values = [('a', 1), ('a' * 44, 1), ('pass', 1), ('asdf£', 1)]
        self.assertEqual(list(f.filter(values)), [('pass', 1)])
        f.finish()
        self.assertEqual(list(f.filter(values)), [('pass', 1)])

    def test_filter_small(self):
        f = pwd_guess.Filterer(pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3))
        self.assertEqual([('aaa', 1)], list(f.filter([('aaa', 1)])))

    def test_filter_freqs(self):
        with tempfile.NamedTemporaryFile() as tf:
            config = pwd_guess.ModelDefaults(
                rare_character_lowest_threshold = 2, char_bag = 'pass\n',
                intermediate_fname = tf.name,
                rare_character_optimization = True)
            f = pwd_guess.Filterer(config)
            self.assertEqual([('pass', 1)],
                             list(f.filter([('asdf£jfj', 1), ('pass', 1)])))
            self.assertEqual({'p' : 1, 'a' : 1, 's': 2}, dict(f.frequencies))
            f.finish()
            self.assertEqual(
                set(config.get_intermediate_info('rare_character_bag')),
                set(['a', 'p']))

    def test_filter_freqs_non_appearing_rare_characters(self):
        with tempfile.NamedTemporaryFile() as tf:
            config = pwd_guess.ModelDefaults(
                rare_character_lowest_threshold = 2, char_bag = 'pass12\n',
                intermediate_fname = tf.name,
                rare_character_optimization = True)
            f = pwd_guess.Filterer(config)
            self.assertEqual([('pass', 1)],
                             list(f.filter([('asdf£jfj', 1), ('pass', 1)])))
            self.assertEqual({'p' : 1, 'a' : 1, 's': 2}, dict(f.frequencies))
            f.finish()
            self.assertEqual(
                set(config.get_intermediate_info('rare_character_bag')),
                set(['1', '2']))

    def test_filter_dict(self):
        f = pwd_guess.Filterer(pwd_guess.ModelDefaults(max_len = 6))
        self.assertEqual([('pass', 1)],
                         list(f.filter([('pass', 1), ('passssss', 2)])))

class ModelSerializerTest(unittest.TestCase):
    def test_model_serializer(self):
        mock = Mock()
        write_value = '{}'
        mock.to_json = MagicMock(return_value = write_value)
        mock.save_weights = MagicMock()
        mock.load_weights = MagicMock()
        with tempfile.NamedTemporaryFile() as fp:
            with tempfile.NamedTemporaryFile() as tp:
                serializer = pwd_guess.ModelSerializer(fp.name, tp.name)
                serializer.save_model(mock)
                self.assertEqual(write_value, fp.read().decode('utf8'))
                serializer = pwd_guess.ModelSerializer(fp.name, tp.name)
                serializer.model_creator_from_json = MagicMock(
                    return_value = mock)
                serializer.load_model()

class GuesserTest(unittest.TestCase):
    def mock_model(self, config, distribution):
        def smart_mock_predict(str_list, **kwargs):
            answer = []
            for i in range(len(str_list)):
                answer.append([distribution.copy()])
            return answer
        mock_model = Mock()
        mock_model.predict = smart_mock_predict
        return mock_model

    def make(self, config, distribution):
        ostream = io.StringIO()
        guesser = pwd_guess.Guesser(self.mock_model(config, distribution),
                                    config, ostream)
        return guesser, ostream

    def test_guesser(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'a\n',
            lower_probability_threshold = 10**-2,
            relevel_not_matching_passwords = False)
        guesser, ostream = self.make(config, [0.5, 0.5])
        guesser.guess()
        self.assertEqual("""	0.5
a	0.25
aa	0.125
aaa	0.0625
""", ostream.getvalue())

    def test_guesser_small_batch(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'abcd\n',
            lower_probability_threshold = 10**-3,
            max_gpu_prediction_size = 3,
            relevel_not_matching_passwords = False)
        guesser, ostream = self.make(config, [0.5, 0.1, 0.1, 0.1, 0.2])
        guesser.guess()

    def test_guesser_small_chunk(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'a\n',
            lower_probability_threshold = 10**-2,
            relevel_not_matching_passwords = False,
            chunk_size_guesser = 2)
        guesser, ostream = self.make(config, [0.5, 0.5])
        guesser.guess()
        self.assertEqual("""	0.5
a	0.25
aa	0.125
aaa	0.0625
""", ostream.getvalue())

    def test_guesser_bigger_rare_c(self):
        with tempfile.NamedTemporaryFile() as intermediatef:
            config = pwd_guess.ModelDefaults(
                parallel_guessing = False, char_bag = 'abAB\n', min_len = 3,
                max_len = 5, uppercase_character_optimization = True,
                random_walk_seed_num = 10000,
                intermediate_fname = intermediatef.name,
                relevel_not_matching_passwords = True,
                rare_character_optimization_guessing = True,
                guess_serialization_method = 'human')
            config.set_intermediate_info(
                'rare_character_bag', [])
            freqs = {
                'a' : .4, 'b' : .4, 'A' : .1, 'B' : .1,
            }
            config.set_intermediate_info('character_frequencies', freqs)
            config.set_intermediate_info(
                'beginning_character_frequencies', freqs)
            config.set_intermediate_info(
                'end_character_frequencies', freqs)
            self.assertTrue(pwd_guess.Filterer(config).pwd_is_valid('aaa'))
            builder = pwd_guess.GuesserBuilder(config)
            mock_model = Mock()
            mock_model.predict = mock_predict_smart_parallel_skewed
            ostream = io.StringIO()
            builder.add_model(mock_model).add_stream(ostream)
            guesser = builder.build()
            guesser.guess()
        found = list(csv.reader(io.StringIO(
            ostream.getvalue()), delimiter = '\t', quotechar = None))
        found = sorted(found, key = lambda row: (float(row[1]), row[0]),
                       reverse = True)
        with open('test_data/test_skewed.sorted.txt', 'r') as expected_data:
            expected = csv.reader(
                expected_data, delimiter = '\t', quotechar = None)
            for i, row in enumerate(expected):
                for item in range(len(row)):
                    value = row[item]
                    try:
                        self.assertAlmostEqual(float(row[item]),
                                               float(found[i][item]))
                    except ValueError:
                        self.assertEqual(row[item], found[i][item])

    def test_guessing_with_relevel(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'a\n',
            lower_probability_threshold = 10**-1)
        guesser, ostream = self.make(config, [0.5, 0.5])
        guesser.guess()
        self.assertEqual("""aaa	1.0
""", ostream.getvalue())

    def test_predict(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'a\n',
            lower_probability_threshold = 10**-1,
            relevel_not_matching_passwords = False)
        guesser, ostream = self.make(config, [0.5, 0.5])
        np.testing.assert_array_equal(
            [0.5, 0.5], guesser.conditional_probs(''))
        np.testing.assert_array_equal(
            [0.5, 0.5], guesser.conditional_probs('a'))
        np.testing.assert_array_equal(
            [0.5, 0.5], guesser.conditional_probs('aa'))
        np.testing.assert_array_equal(
            [0.5, 0.5], guesser.conditional_probs('aaa'))

    def test_relevel(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'a\n',
            lower_probability_threshold = 10**-1)
        guesser, ostream = self.make(config, [0.5, 0.5])
        np.testing.assert_array_equal([0.0, 1.0], guesser.conditional_probs(''))
        np.testing.assert_array_equal(
            [0.0, 1.0], guesser.conditional_probs('a'))
        np.testing.assert_array_equal(
            [0.0, 1.0], guesser.conditional_probs('aa'))
        np.testing.assert_array_equal(
            [1.0, 0.0], guesser.conditional_probs('aaa'))

    def test_relevel_tri_alpha(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'ab\n',
            lower_probability_threshold = 10**-1)
        guesser, ostream = self.make(config, [0.5, 0.2, 0.3])
        np.testing.assert_array_equal(
            [0.0, .4, .6], guesser.conditional_probs(''))
        np.testing.assert_array_equal(
            [0.0, .4, .6], guesser.conditional_probs('a'))
        np.testing.assert_array_equal(
            [0.0, .4, .6], guesser.conditional_probs('aa'))
        np.testing.assert_array_equal(
            [0.0, .4, .6], guesser.conditional_probs('ab'))
        np.testing.assert_array_equal(
            [1.0, 0.0, 0.0], guesser.conditional_probs('aaa'))

    def test_relevel_tri_alpha_calculator(self):
        distribution = [0.5, 0.2, 0.3]
        def smart_mock_predict(str_list, **kwargs):
            answer = []
            for i in range(len(str_list)):
                answer.append([distribution.copy()])
            return answer
        with tempfile.NamedTemporaryFile(mode = 'w') as pwd_file, \
             tempfile.NamedTemporaryFile(mode = 'r') as gfile:
            password_list = ['aaa', 'abb', 'aab']
            for pwd in password_list:
                pwd_file.write('%s\n' % pwd)
            pwd_file.flush()
            config = pwd_guess.ModelDefaults(
                min_len = 3, max_len = 3, char_bag = 'ab\n',
                lower_probability_threshold = 10**-2,
                guess_serialization_method = 'calculator',
                password_test_fname = pwd_file.name)
            mock_model = Mock()
            mock_model.predict = smart_mock_predict
            guesser = (pwd_guess.GuesserBuilder(config)
                       .add_model(mock_model).add_file(gfile.name).build())
            self.assertEqual(type(guesser.output_serializer),
                             pwd_guess.GuessNumberGenerator)
            np.testing.assert_array_equal(
                [0.0, .4, .6], guesser.conditional_probs(''))
            np.testing.assert_array_equal(
                [0.0, .4, .6], guesser.conditional_probs('a'))
            np.testing.assert_array_equal(
                [0.0, .4, .6], guesser.conditional_probs('aa'))
            np.testing.assert_array_equal(
                [0.0, .4, .6], guesser.conditional_probs('ab'))
            np.testing.assert_array_equal(
                [1.0, 0.0, 0.0], guesser.conditional_probs('aaa'))
            guesser.complete_guessing()
            # IMPORTANT: This distribution should give a guess count of 4 for
            # baa. However, due to floating point rounding error in python, this
            # is not the case. It seems that in python,
            # .4 * .4 * .6 != .6 * .4 * .4
            # This example is fairly sensitive to floating point rounding error
            # The GPU also computes at a rounding error higher than the original
            # computation
            self.assertEqual("""Total count: 8
abb	0.144	1
aab	0.096	4
aaa	0.064	7
""", gfile.read())

    def test_do_guessing(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'a\n',
            lower_probability_threshold = 10**-2,
            relevel_not_matching_passwords = False)
        model = self.mock_model(config, [0.5, 0.5])
        with tempfile.NamedTemporaryFile() as fp:
            (pwd_guess.GuesserBuilder(config).add_model(model).add_file(fp.name)
             .build().complete_guessing())
            self.assertEqual("""	0.5
a	0.25
aa	0.125
aaa	0.0625
""", fp.read().decode('utf8'))

def mock_predict_smart_parallel(input_vec, **kwargs):
    answer = []
    for i in range(len(input_vec)):
        answer.append([[0.5, 0.25, 0.25].copy()])
    return answer

def mock_predict_smart_parallel_skewed(input_vec, **kwargs):
    answer = []
    for i in range(len(input_vec)):
        answer.append([[0.5, 0.1, 0.4].copy()])
    return answer

class ParallelGuesserTest(unittest.TestCase):
    mock_model = Mock()

    def setUp(self):
        self.mock_model.predict = mock_predict_smart_parallel
        self.intermediate_dir = tempfile.mkdtemp()
        self.config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'ab\n', fork_length = 2,
            guesser_intermediate_directory = self.intermediate_dir)
        self.mock_output = io.StringIO()
        self.archfile = tempfile.NamedTemporaryFile(mode = 'w', delete = False)
        self.weightfile = tempfile.NamedTemporaryFile(
            mode = 'w', delete = False)
        self.serializer = Mock()
        self.serializer.archfile = self.archfile.name
        self.serializer.weightfile = self.weightfile.name
        self.serializer.load_model = MagicMock(return_value = self.mock_model)

    def tearDown(self):
        shutil.rmtree(self.intermediate_dir)
        os.remove(self.archfile.name)
        os.remove(self.weightfile.name)

    def test_get_fork_points(self):
        config_dict = self.config.as_dict()
        config_dict['relevel_not_matching_passwords'] = False
        parallel_guesser = pwd_guess.ParallelGuesser(
            self.serializer, pwd_guess.ModelDefaults(**config_dict),
            self.mock_output)
        parallel_guesser.fork_starter = MagicMock(return_value = None)
        parallel_guesser.recur('', 1)
        self.assertEqual([('aa', .0625),
                          ('ab', .0625),
                          ('ba', .0625),
                          ('bb', .0625)], parallel_guesser.fork_points)

    def test_forking(self):
        parallel_guesser = pwd_guess.ParallelGuesser(
            self.serializer, self.config, self.mock_output)
        json.dump({
            'mock_model' : [0.5, 0.25, 0.25]
        }, self.archfile)
        self.archfile.flush()
        parallel_guesser.guess()
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(io.StringIO(self.mock_output.getvalue()),
                               delimiter = '\t')]
        sort_freq = sorted(pwd_freq, key = lambda x: x[0])
        self.assertEqual([('aaa', .125), ('aab', .125),
                          ('aba', .125), ('abb', .125),
                          ('baa', .125), ('bab', .125),
                          ('bba', .125), ('bbb', .125)], sort_freq)
        self.assertEqual(8, parallel_guesser.generated)

    def test_forking_calculator(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as tf:
            for s in ['aaa', 'bbb', 'aab']:
                tf.write('%s\n' % (s))
            tf.flush()
            self.config.guess_serialization_method = 'calculator'
            self.config.password_test_fname = tf.name
            parallel_guesser = pwd_guess.ParallelGuesser(
                self.serializer, self.config, self.mock_output,
            )
            json.dump({
                'mock_model' : [0.5, 0.25, 0.25]
            }, self.archfile)
            self.archfile.flush()
            parallel_guesser.guess()
            odata_file = io.StringIO(self.mock_output.getvalue())
            line = odata_file.readline()
            out_data = list(csv.reader(odata_file, delimiter = '\t'))
            pwd_freq = [(row[0], float(row[1]), int(row[2]))
                        for row in out_data]
            sort_freq = sorted(pwd_freq, key = lambda x: x[0])
            self.assertEqual([('aaa', .125, 0), ('aab', .125, 0),
                              ('bbb', .125, 0)], sort_freq)
            self.assertEqual(8, parallel_guesser.generated)

    def test_parse_cmd(self):
        cmd = pwd_guess.ParallelGuesser.subp_command('argfname', 'logfile')
        self.assertEqual(cmd[2:], [
            '--forked', 'guesser', '--config-args', 'argfname', '--log-file',
            'logfile'])
        pwd_guess.make_parser().parse_args(cmd[2:])

    def test_map_pool(self):
        pg = pwd_guess.ParallelGuesser(
            self.serializer, self.config, self.mock_output)
        pg.map_pool([(
            self.config.as_dict(), ['na', 'na'], ['aa', 0.125])], 1, 2)

class GuesserBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tempf = tempfile.NamedTemporaryFile()

    def tearDown(self):
        if self.tempf is not None:
            self.tempf.close()

    def test_create(self):
        pwd_guess.GuesserBuilder(pwd_guess.ModelDefaults())

    def test_make_simple_guesser(self):
        builder = pwd_guess.GuesserBuilder(
            pwd_guess.ModelDefaults(parallel_guessing = False))
        mock_model, mock_stream = Mock(), Mock()
        builder.add_model(mock_model).add_stream(mock_stream)
        guesser = builder.build()
        self.assertNotEqual(guesser, None)
        self.assertEqual(guesser.model, mock_model)
        self.assertEqual(guesser.output_serializer.ostream, mock_stream)

    def test_make_parallel_guesser(self):
        builder = pwd_guess.GuesserBuilder(
            pwd_guess.ModelDefaults(parallel_guessing = True))
        mock_serializer, mock_stream, mock_model = Mock(), Mock(), Mock()
        mock_serializer.load_model = MagicMock(return_value = mock_model)
        builder.add_serializer(mock_serializer).add_stream(mock_stream)
        guesser = builder.build()
        self.assertNotEqual(guesser, None)
        self.assertEqual(guesser.model, mock_model)
        self.assertEqual(guesser.real_output, mock_stream)

    def test_make_simple_guesser_file(self):
        builder = pwd_guess.GuesserBuilder(
            pwd_guess.ModelDefaults(parallel_guessing = False))
        mock_model = Mock()
        builder.add_model(mock_model).add_file(self.tempf.name)
        guesser = builder.build()
        self.assertNotEqual(guesser, None)
        self.assertNotEqual(guesser.output_serializer.ostream, None)
        self.assertEqual(guesser.model, mock_model)

    def test_make_simple_guesser(self):
        builder = pwd_guess.GuesserBuilder(
            pwd_guess.ModelDefaults(parallel_guessing = True))
        mock_serializer, mock_stream, mock_model = Mock(), Mock(), Mock()
        mock_serializer.load_model = MagicMock(return_value = mock_model)
        builder.add_serializer(mock_serializer).add_stream(mock_stream)
        builder.add_parallel_setting(False)
        guesser = builder.build()
        self.assertNotEqual(guesser, None)
        self.assertEqual(type(guesser), pwd_guess.Guesser)

class PreprocessingStepTest(unittest.TestCase):
    base_config = {
        "training_chunk" : 64,
        "layers" : 3,
        "hidden_size" : 128,
        "generations" : 1,
        "min_len" : 3,
        "max_len" : 5,
        "char_bag" : "ab\n",
        "training_accuracy_threshold": -1,
        "trie_fname" : ":memory:"
    }

    def setUp(self):
        self.input_file = tempfile.NamedTemporaryFile(mode = 'w')

    def tearDown(self):
        self.input_file.close()

    def do_preprocessing(self, config):
        real_config_dict = self.base_config.copy()
        real_config_dict.update(config)
        real_config = pwd_guess.ModelDefaults(real_config_dict)
        self.real_config = real_config
        self.input_file.write("""aaaa\t2
abbbb\t4
abab\t1
aaab\t3""")
        self.input_file.flush()
        plist = pwd_guess.read_passwords(
            [self.input_file.name], ['tsv'], real_config)
        preprocessor = pwd_guess.BasePreprocessor.fromConfig(real_config)
        preprocessor.begin(plist)
        return preprocessor.stats()

    def test_normal(self):
        self.assertEqual(self.do_preprocessing({
            'simulated_frequency_optimization' : False
        }), 54)

    def test_simulated_freq(self):
        self.assertEqual(self.do_preprocessing({
            'simulated_frequency_optimization' : True
        }), 21)

    def test_trie_reg(self):
        self.assertEqual(self.do_preprocessing({
            'simulated_frequency_optimization' : True,
            'trie_implementation' : 'trie'
        }), 15)
        self.assertFalse(os.path.exists(":memory:"))

    def test_trie_super(self):
        self.assertEqual(self.do_preprocessing({
            'simulated_frequency_optimization' : True,
            'trie_implementation' : 'trie',
            'trie_serializer_type' : 'fuzzy'
        }), 12)
        self.assertFalse(os.path.exists(":memory:"))

    def test_trie_fuzzy_disk(self):
        self.assertFalse(os.path.exists('trie_storage'))
        try:
            self.assertEqual(self.do_preprocessing({
                'simulated_frequency_optimization' : True,
                'trie_implementation' : 'disk',
                'trie_serializer_type' : 'fuzzy',
                'trie_fname' : 'trie_storage',
                'trie_intermediate_storage' : 'trie_intermediate'
            }), 12)
            self.assertFalse(os.path.exists(":memory:"))
            pre = pwd_guess.DiskPreprocessor(self.real_config)
            pre.begin()
            self.assertEqual(12, pre.stats())
        finally:
            shutil.rmtree('trie_intermediate')
            os.remove('trie_storage')

    def test_trie_fuzzy_disk_intermediate(self):
        self.assertFalse(os.path.exists('trie_storage'))
        self.assertFalse(os.path.exists('trie_intermediate'))
        self.assertFalse(os.path.exists('intermediate_data.sqlite'))
        try:
            self.assertEqual(self.do_preprocessing({
                'simulated_frequency_optimization' : True,
                'trie_implementation' : 'disk',
                'trie_serializer_type' : 'fuzzy',
                'trie_fname' : 'trie_storage',
                'trie_intermediate_storage' : 'trie_intermediate',
                'intermediate_fname' : 'intermediate_data.sqlite',
                'preprocess_trie_on_disk' : True,
                'rare_character_optimization' : True,
                'rare_character_lowest_threshold' : 1,
                'char_bag' : 'abc\n'
            }), 12)
            self.assertFalse(os.path.exists(":memory:"))
            pre = pwd_guess.BasePreprocessor.byFormat('im_trie',
                                                      self.real_config)
            pre.begin()
            self.assertEqual(type(pre), pwd_guess.IntermediatePreprocessor)
            self.assertEqual(12, pre.stats())
        finally:
            shutil.rmtree('trie_intermediate')
            os.remove('trie_storage')
            os.remove('intermediate_data.sqlite')

class ProbabilityCalculatorTest(unittest.TestCase):
    def test_calc_one(self):
        mock_guesser = Mock()
        mock_guesser.config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'ab\n',
            relevel_not_matching_passwords = False)
        mock_guesser.batch_prob = MagicMock(
            return_value=[[[0, 0.5, 0.5]],
                          [[0, 0.5, 0.5]],
                          [[0, 0.5, 0.5]],
                          [[1, 0, 0]]])
        p = pwd_guess.ProbabilityCalculator(mock_guesser)
        self.assertEqual(list(p.calc_probabilities([('aaa', 1)])),
                         [('aaa', 0.125)])

    def test_calc_two(self):
        mock_guesser = Mock()
        mock_guesser.config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'ab\n',
            relevel_not_matching_passwords = False)
        mock_guesser.batch_prob = MagicMock(
            return_value=[[[0, 0.5, 0.5]],
                          [[0, 0.5, 0.5]],
                          [[0, 0.5, 0.5]],
                          [[1, 0, 0]],
                          [[0, 0.5, 0.5]],
                          [[0, 0.5, 0.5]],
                          [[0, 0.5, 0.5]],
                          [[1, 0, 0]]])
        p = pwd_guess.ProbabilityCalculator(mock_guesser)
        self.assertEqual(set(p.calc_probabilities([('aaa', 1), ('abb', 1)])),
                         set([('aaa', 0.125), ('abb', 0.125)]))

    def test_calc_prefix(self):
        mock_guesser = Mock()
        mock_guesser.config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'ab\n',
            relevel_not_matching_passwords = False)
        mock_guesser.batch_prob = MagicMock(
            return_value=[[[0, 0.5, 0.5]],
                          [[0, 0.4, 0.6]],
                          [[0, 0.5, 0.5]],
                          [[.5, .25, .25]],
                          [[0, 0.5, 0.5]],
                          [[0, 0.4, 0.6]],
                          [[0, 0.5, 0.5]],
                          [[.5, .25, .25]]])
        p = pwd_guess.ProbabilityCalculator(mock_guesser, prefixes = True)
        self.assertEqual(set(p.calc_probabilities([('aaa', 1), ('abb', 1)])),
                         set([('aaa', 0.1), ('abb', 0.15)]))

    def test_calc_optimizing(self):
        with tempfile.NamedTemporaryFile() as tf:
            config = pwd_guess.ModelDefaults(
                min_len = 2, max_len = 2, char_bag = 'abAB\n',
                uppercase_character_optimization = True,
                rare_character_optimization = False,
                relevel_not_matching_passwords = False,
                lower_probability_threshold = .1,
                intermediate_fname = tf.name)
            config.set_intermediate_info('rare_character_bag', [])
            freqs = {
                'a' : .3, 'b' : .4, 'A' : .2, 'B' : .1
            }
            config.set_intermediate_info('character_frequencies', freqs)
            config.set_intermediate_info(
                'beginning_character_frequencies', freqs)
            config.set_intermediate_info(
                'end_character_frequencies', freqs)
            mock_guesser = Mock()
            mock_guesser.config = config
            mock_guesser.batch_prob = MagicMock(
                return_value=[[[0, 0.5, 0.5]],
                              [[0, 0.5, 0.5]],
                              [[1, 0, 0]],
                              [[0, 0.5, 0.5]],
                              [[0, 0.5, 0.5]],
                              [[1, 0, 0]]])
            p = pwd_guess.ProbabilityCalculator(mock_guesser)
            self.assertEqual(set(p.calc_probabilities(
                [('aa', 1), ('bB', 1)])),
                             set([('aa', .09), ('bB', 0.04000000000000001)]))

class GuessNumberGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.ostream = tempfile.NamedTemporaryFile(mode = 'w', delete = False)

    def tearDown(self):
        os.remove(self.ostream.name)

    def test_guessing(self):
        probs = [('pass', .025), ('word', .25), ('gmail', .04)]
        gng = pwd_guess.GuessNumberGenerator(self.ostream, probs)
        gng.serialize('asdf', .3)
        gng.serialize('word', .25)
        gng.serialize('gmail', .04)
        gng.serialize('pass', .025)
        gng.serialize('jjjj', .2)
        gng.serialize('jjjjj', .00001)
        gng.finish()
        with open(self.ostream.name, 'r') as guesses:
            self.assertEqual(
                'Total count: 6\nword\t0.25\t1\ngmail\t0.04\t3\npass\t0.025\t4\n',
                guesses.read())

    def test_guessing_real(self):
        probs = [('    ', 1.26799704013e-05), ('william', 2.12144662517e-05),
                 ('forever', 0.00013370734607), ('8daddy', 1.00234253381e-05)]
        gng = pwd_guess.GuessNumberGenerator(self.ostream, probs)
        with open('test_data/guesses.test.data.txt', 'r') as test_data:
            for row in csv.reader(
                    test_data, delimiter = '\t', quotechar = None):
                gng.serialize(row[0], float(row[1]))
        gng.finish()
        with open(self.ostream.name, 'r') as guesses:
            self.assertEqual(
                """Total count: 235
forever	0.00013370734607	0
william	2.12144662517e-05	72
    	1.26799704013e-05	160
8daddy	1.00234253381e-05	234
""",
                guesses.read())

class PasswordTemplateSerializerTest(unittest.TestCase):
    def test_serialize(self):
        serialized = []
        self.finished = False
        def mock_serialize(pwd, prob):
            serialized.append((pwd, prob))
        def mock_finish():
            self.finished = True
        mock_serializer = Mock()
        mock_serializer.serialize = mock_serialize
        mock_serializer.finish = mock_finish
        with tempfile.NamedTemporaryFile() as tf:
            config = pwd_guess.ModelDefaults(
                min_len = 2, max_len = 2, char_bag = 'abABc:!@\n',
                uppercase_character_optimization = True,
                rare_character_optimization = True,
                relevel_not_matching_passwords = False,
                lower_probability_threshold = .007,
                intermediate_fname = tf.name)
            config.set_intermediate_info(
                'rare_character_bag', [':', '!', '@'])
            freqs = {
                'a' : .2, 'b' : .2, 'A' : .1, 'B' : .1,
                ':' : .19, '!' : .19, '@' : .02
            }
            config.set_intermediate_info('character_frequencies', freqs)
            config.set_intermediate_info(
                'beginning_character_frequencies', freqs)
            end_freqs = freqs.copy()
            end_freqs[':'] = .3
            end_freqs['!'] = .1
            end_freqs['@'] = 0
            config.set_intermediate_info(
                'end_character_frequencies', end_freqs)
            pts = pwd_guess.PasswordTemplateSerializer(config, mock_serializer)
            self.assertAlmostEqual(pts.calc('a', 'A'), .33333333)
            self.assertAlmostEqual(pts.calc('a', 'a'), .66666666)
            self.assertAlmostEqual(pts.calc(':', '!'), .475)
            pts.serialize('aa', .5)
            pts.serialize('b:', .4)
            pts.serialize('cc', .4)
            self.assertAlmostEqual(
                pts.find_real_pwd('aa', 'aA'), (2/3) * (1/3))
            self.assertAlmostEqual(
                pts.find_real_pwd('aa', 'aa'), (2/3) * (2/3))
            self.assertAlmostEqual(
                pts.find_real_pwd('b:', 'b:'), (2/3) * (.3/.4))
            self.assertFalse(self.finished)
            pts.finish()
            self.assertTrue(self.finished)
        self.assertEqual(set(serialized), set([
            ('cc', .4),
            ('aa', .5 * (2/3) * (2/3)), ('aA', .5 * (2/3) * (1/3)),
            ('Aa', .5 * (1/3) * (2/3)), ('AA', .5 * (1/3) * (1/3)),
            ('b:', .4 * (2/3) * (.3 / .4)),
            ('B:', .4 * (1/3) * (.3 / .4)),
            ('b!', .4 * (2/3) * (.1 / .4)),
            ('B!', .4 * (1/3) * (.1 / .4))]))

    def test_expand(self):
        with tempfile.NamedTemporaryFile() as tf:
            config = pwd_guess.ModelDefaults(
                min_len = 3, max_len = 3, char_bag = 'abAB:^\n',
                relevel_not_matching_passwords = False,
                uppercase_character_optimization = True,
                rare_character_optimization = True,
                intermediate_fname = tf.name)
            config.set_intermediate_info(
                'rare_character_bag', [':', '^'])
            freqs = {
                'a' : .2, 'b' : .2, 'A' : .1, 'B' : .1, '^' : .2, ':' : .2
            }
            config.set_intermediate_info(
                'character_frequencies', freqs)
            config.set_intermediate_info(
                'beginning_character_frequencies', freqs)
            config.set_intermediate_info(
                'end_character_frequencies', freqs)
            pts = pwd_guess.PasswordTemplateSerializer(config, Mock())
            np.testing.assert_array_equal(
                pts.expand_conditional_probs(np.array([.2, .1, .2, .5]), ''), [
                    0.13333333333333333, 0.3333333333333333,
                    0.06666666666666667, 0.16666666666666666, .05, .05, .2])

class RandomWalkGuesserTest(unittest.TestCase):
    def setUp(self):
        self.tempf = tempfile.NamedTemporaryFile(delete = False)

    def tearDown(self):
        self.tempf.close()
        if os.path.exists(self.tempf.name):
            os.remove(self.tempf.name)

    def test_create(self):
        builder = pwd_guess.GuesserBuilder(pwd_guess.ModelDefaults(
            parallel_guessing = False,
            guess_serialization_method = 'random_walk'))
        mock_model = Mock()
        mock_model.predict = mock_predict_smart_parallel
        builder.add_model(mock_model).add_file(self.tempf.name)
        guesser = builder.build()
        self.assertEqual(pwd_guess.RandomWalkGuesser, type(guesser))

    def test_seed_data(self):
        config = pwd_guess.ModelDefaults(
            parallel_guessing = False,
            char_bag = 'ab\n',
            random_walk_seed_num = 2,
            guess_serialization_method = 'random_walk')
        builder = pwd_guess.GuesserBuilder(config)
        mock_model = Mock()
        mock_model.predict = mock_predict_smart_parallel
        builder.add_model(mock_model).add_file(self.tempf.name)
        guesser = builder.build()
        g = list(guesser.seed_data())
        self.assertEqual(g, [('', 1, 1, 0), ('', 1, 1, 0)])

    def test_next_nodes(self):
        config = pwd_guess.ModelDefaults(
            parallel_guessing = False,
            char_bag = 'ab\n',
            guess_serialization_method = 'random_walk')
        builder = pwd_guess.GuesserBuilder(config)
        mock_model = Mock()
        mock_model.predict = mock_predict_smart_parallel
        builder.add_model(mock_model).add_file(self.tempf.name)
        guesser = builder.build()
        next = generator.next_nodes_random_walk(
            guesser, 'aa', .5, np.array([.5, .25, .25]))
        self.assertEqual(list(next), [
            ('aa\n', .25, .5), ('aaa', .125, .25), ('aab', .125, .25)])

    def test_guess(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as gf:
            gf.write('aaa\nbbb\n')
            gf.flush()
            pw = pwd_guess.PwdList(gf.name)
            self.assertEqual(list(pw.as_list()), [('aaa', 1), ('bbb', 1)])
            config = pwd_guess.ModelDefaults(
                parallel_guessing = False, char_bag = 'ab\n', min_len = 3,
                max_len = 3, password_test_fname = gf.name,
                random_walk_seed_num = 1000,
                relevel_not_matching_passwords = True,
                guess_serialization_method = 'random_walk')
            self.assertTrue(pwd_guess.Filterer(config).pwd_is_valid('aaa'))
            builder = pwd_guess.GuesserBuilder(config)
            mock_model = Mock()
            mock_model.predict = mock_predict_smart_parallel_skewed
            builder.add_model(mock_model).add_file(self.tempf.name)
            guesser = builder.build()
            guesser.complete_guessing()
            with open(self.tempf.name, 'r') as output:
                reader = list(csv.reader(
                    output, delimiter = '\t', quotechar = None))
                self.assertEqual(len(reader), 2)
                for row in reader:
                    pwd, prob, gn, *_ = row
                    self.assertTrue(pwd == 'aaa' or pwd == 'bbb')
                    self.assertEqual(prob, '0.008' if pwd == 'aaa' else '0.512')
                    self.assertAlmostEqual(float(gn), 8 if pwd == 'aaa' else 1, delta = 2)

    def test_guess_wide(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as gf:
            gf.write('aaaa\nbbbba\n')
            gf.flush()
            pw = pwd_guess.PwdList(gf.name)
            self.assertEqual(list(pw.as_list()), [('aaaa', 1), ('bbbba', 1)])
            config = pwd_guess.ModelDefaults(
                parallel_guessing = False, char_bag = 'ab\n', min_len = 3,
                max_len = 5, password_test_fname = gf.name,
                random_walk_seed_num = 10000,
                relevel_not_matching_passwords = True,
                guess_serialization_method = 'random_walk')
            self.assertTrue(pwd_guess.Filterer(config).pwd_is_valid('aaa'))
            builder = pwd_guess.GuesserBuilder(config)
            mock_model = Mock()
            mock_model.predict = mock_predict_smart_parallel_skewed
            builder.add_model(mock_model).add_file(self.tempf.name)
            guesser = builder.build()
            guesser.complete_guessing()
            with open(self.tempf.name, 'r') as output:
                reader = list(csv.reader(
                    output, delimiter = '\t', quotechar = None))
                self.assertEqual(len(reader), 2)
                for row in reader:
                    pwd, prob, gn, *_ = row
                    self.assertTrue(pwd == 'aaaa' or pwd == 'bbbba')
                    self.assertEqual(prob, '0.0004' if pwd == 'aaaa' else '0.02048')
                    self.assertAlmostEqual(float(gn), 50 if pwd == 'aaaa' else 15, delta = 2)

    def test_guess_simulated(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as gf, \
             tempfile.NamedTemporaryFile() as intermediatef:
            gf.write('aaaa\nbbbBa\n')
            gf.flush()
            pw = pwd_guess.PwdList(gf.name)
            self.assertEqual(list(pw.as_list()), [('aaaa', 1), ('bbbBa', 1)])
            config = pwd_guess.ModelDefaults(
                parallel_guessing = False, char_bag = 'abAB\n', min_len = 3,
                max_len = 5, password_test_fname = gf.name,
                uppercase_character_optimization = True,
                random_walk_seed_num = 10,
                random_walk_upper_bound = 1000,
                rare_character_optimization_guessing = True,
                intermediate_fname = intermediatef.name,
                relevel_not_matching_passwords = True,
                guess_serialization_method = 'random_walk')
            config.set_intermediate_info(
                'rare_character_bag', [])
            freqs = {
                'a' : .4, 'b' : .4, 'A' : .1, 'B' : .1,
            }
            config.set_intermediate_info('character_frequencies', freqs)
            config.set_intermediate_info(
                'beginning_character_frequencies', freqs)
            config.set_intermediate_info(
                'end_character_frequencies', freqs)
            self.assertTrue(pwd_guess.Filterer(config).pwd_is_valid('aaa'))
            builder = pwd_guess.GuesserBuilder(config)
            mock_model = Mock()
            mock_model.predict = mock_predict_smart_parallel_skewed
            builder.add_model(mock_model).add_file(self.tempf.name)
            guesser = builder.build()
            guesser.complete_guessing()
            with open(self.tempf.name, 'r') as output:
                reader = list(csv.reader(
                    output, delimiter = '\t', quotechar = None))
                self.assertEqual(len(reader), 2)
                for row in reader:
                    pwd, prob, gn, *_ = row
                    self.assertTrue(pwd == 'aaaa' or pwd == 'bbbBa')
                    self.assertEqual(prob, '0.00016384' if pwd == 'aaaa' else '0.0016777216')
                    self.assertAlmostEqual(float(gn), 397 if pwd == 'aaaa' else 137, delta = 20)

class ParallelRandomWalkGuesserTest(unittest.TestCase):
    def setUp(self):
        self.tempf = tempfile.NamedTemporaryFile(delete = False)
        self.intermediate_dir = tempfile.mkdtemp()
        self.archfile = tempfile.NamedTemporaryFile(mode = 'w', delete = False)
        self.weightfile = tempfile.NamedTemporaryFile(
            mode = 'w', delete = False)
        self.serializer = Mock()
        self.serializer.archfile = self.archfile.name
        self.serializer.weightfile = self.weightfile.name
        mock_model = Mock()
        mock_model.predict = mock_predict_smart_parallel_skewed
        self.serializer.load_model = MagicMock(return_value = mock_model)

    def tearDown(self):
        self.tempf.close()
        if os.path.exists(self.tempf.name):
            os.remove(self.tempf.name)
        shutil.rmtree(self.intermediate_dir)
        os.remove(self.archfile.name)
        os.remove(self.weightfile.name)

    def test_arglist(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as gf, \
             tempfile.NamedTemporaryFile() as intermediatef:
            gf.write('aaaa\nbbbBa\n')
            gf.flush()
            config = pwd_guess.ModelDefaults(
                char_bag = 'abAB\n', min_len = 3, max_len = 5,
                uppercase_character_optimization = True,
                rare_character_optimization_guessing = True,
                rare_character_optimization = True,
                intermediate_fname = intermediatef.name,
                parallel_guessing = True, password_test_fname = gf.name,
                guess_serialization_method = 'random_walk', cpu_limit = 2)
            builder = pwd_guess.GuesserBuilder(config)
            json.dump({
                'mock_model' : [0.5, 0.1, 0.4]
            }, self.archfile)
            self.archfile.flush()
            config.set_intermediate_info(
                'rare_character_bag', [])
            freqs = {
                'a' : .4, 'b' : .4, 'A' : .1, 'B' : .1,
            }
            config.set_intermediate_info('character_frequencies', freqs)
            config.set_intermediate_info(
                'beginning_character_frequencies', freqs)
            config.set_intermediate_info(
                'end_character_frequencies', freqs)
            builder.add_serializer(self.serializer).add_file(self.tempf.name)
            guesser = builder.build()
            self.assertEqual(type(guesser), pwd_guess.ParallelRandomWalker)
            arg_list = list(guesser.arg_list())
            self.assertEqual(len(arg_list), 2)
            self.assertEqual(set(itertools.chain.from_iterable(arg_list)),
                             set([('bbbBa', 0.0016777216000000014),
                                  ('aaaa', 0.00016384000000000011)]))

    def test_guess_simulated(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as gf, \
             tempfile.NamedTemporaryFile() as intermediatef:
            gf.write('aaaa\nbbbBa\n')
            gf.flush()
            pw = pwd_guess.PwdList(gf.name)
            self.assertEqual(list(pw.as_list()), [('aaaa', 1), ('bbbBa', 1)])
            config = pwd_guess.ModelDefaults(
                parallel_guessing = True, char_bag = 'abAB\n', min_len = 3,
                max_len = 5, password_test_fname = gf.name,
                uppercase_character_optimization = True,
                random_walk_seed_num = 10000,
                rare_character_optimization_guessing = True,
                intermediate_fname = intermediatef.name,
                guesser_intermediate_directory = self.intermediate_dir,
                relevel_not_matching_passwords = True,
                cpu_limit = 2,
                guess_serialization_method = 'random_walk')
            config.set_intermediate_info(
                'rare_character_bag', [])
            freqs = {
                'a' : .4, 'b' : .4, 'A' : .1, 'B' : .1,
            }
            config.set_intermediate_info('character_frequencies', freqs)
            config.set_intermediate_info(
                'beginning_character_frequencies', freqs)
            config.set_intermediate_info(
                'end_character_frequencies', freqs)
            json.dump({
                'mock_model' : [0.5, 0.1, 0.4]
            }, self.archfile)
            self.archfile.flush()
            builder = pwd_guess.GuesserBuilder(config)
            builder.add_serializer(self.serializer).add_file(self.tempf.name)
            guesser = builder.build()
            self.assertTrue(guesser.model is not None)
            guesser.complete_guessing()
            with open(self.tempf.name, 'r') as output:
                reader = list(csv.reader(
                    output, delimiter = '\t', quotechar = None))
                self.assertEqual(len(reader), 2)
                for row in reader:
                    pwd, prob, gn, *_ = row
                    self.assertTrue(pwd == 'aaaa' or pwd == 'bbbBa')
                    self.assertEqual(prob, '0.0001638400000000001' if pwd == 'aaaa' else '0.0016777216000000014')
                    self.assertAlmostEqual(float(gn), 397 if pwd == 'aaaa' else 137, delta = 20)

class PolicyTests(unittest.TestCase):
    def test_basic(self):
        config = Mock()
        config.enforced_policy = 'basic'
        policy = pwd_guess.BasePasswordPolicy.fromConfig(config)
        self.assertTrue(type(policy), pwd_guess.BasicPolicy)
        self.assertTrue(policy.pwd_complies('asdf'))
        self.assertTrue(policy.pwd_complies('asdf' * 30))
        self.assertTrue(policy.pwd_complies(''))

    def test_basic_long(self):
        config = Mock()
        config.enforced_policy = 'basic_long'
        policy = pwd_guess.BasePasswordPolicy.fromConfig(config)
        self.assertTrue(type(policy), pwd_guess.PasswordPolicy)
        self.assertFalse(policy.pwd_complies('asdf'))
        self.assertFalse(policy.pwd_complies('asdfasd'))
        self.assertFalse(policy.pwd_complies(''))
        self.assertTrue(policy.pwd_complies('asdf' * 30))
        self.assertTrue(policy.pwd_complies('asdfasdf'))
        self.assertTrue(policy.pwd_complies('asdfasdfasdfasdf'))

    def test_complex(self):
        config = Mock()
        config.enforced_policy = 'complex'
        policy = pwd_guess.BasePasswordPolicy.fromConfig(config)
        self.assertTrue(type(policy), pwd_guess.ComplexPasswordPolicy)

        self.assertTrue(policy.has_group('asdf', policy.lowercase))
        self.assertFalse(policy.has_group('1', policy.lowercase))
        self.assertTrue(policy.has_group('10', policy.digits))
        self.assertFalse(policy.has_group('a', policy.digits))
        self.assertTrue(policy.has_group('A', policy.uppercase))
        self.assertFalse(policy.has_group('a', policy.uppercase))
        self.assertTrue(policy.all_from_group('asdf0A', policy.non_symbols))
        self.assertFalse(policy.all_from_group('asdf*', policy.non_symbols))
        self.assertTrue(policy.passes_blacklist('asdf*'))

        self.assertFalse(policy.pwd_complies('asdf'))
        self.assertFalse(policy.pwd_complies('asdfasd'))
        self.assertFalse(policy.pwd_complies(''))
        self.assertFalse(policy.pwd_complies('asdf' * 30))
        self.assertFalse(policy.pwd_complies('asdfasdf'))
        self.assertFalse(policy.pwd_complies('asdfasdfasdfasdf'))
        self.assertFalse(policy.pwd_complies('1Aasdfasdfasdfasdf'))
        self.assertFalse(policy.pwd_complies('1Aa*'))
        self.assertFalse(policy.pwd_complies('1A*'))
        self.assertFalse(policy.pwd_complies('111*asdf'))

        self.assertTrue(policy.pwd_complies('1Aasdfasdfasdfasdf*'))
        self.assertTrue(policy.pwd_complies('1Aa*asdf'))
        self.assertTrue(policy.pwd_complies('999Apple*'))
        self.assertTrue(policy.pwd_complies('111*Asdf'))
        self.assertTrue(policy.pwd_complies('111*jjjJ'))

        with tempfile.NamedTemporaryFile(mode = 'w') as temp_bl:
            temp_bl.write('asdf\n')
            temp_bl.write('apple\n')
            temp_bl.flush()
            policy.load_blacklist(temp_bl.name)

        self.assertFalse(policy.pwd_complies('111*Asdf'))
        self.assertFalse(policy.pwd_complies('999Apple*'))
        self.assertTrue(policy.pwd_complies('111*jjjJ'))

if __name__ == '__main__':
    unittest.main()
