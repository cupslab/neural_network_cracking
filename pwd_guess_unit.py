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

import pwd_guess

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

class NodeTrieTest(unittest.TestCase):
    def setUp(self):
        self.trie = pwd_guess.NodeTrie()

    def test_iterate(self):
        self.trie.increment('aaa', 1)
        self.assertEqual([('a', 1), ('aa', 1), ('aaa', 1)],
                         list(self.trie.iterate('reg')))
        self.assertEqual(self.trie.size(), 3)

    def test_iterate_overlap(self):
        self.trie.increment('aaa', 1)
        self.trie.increment('aab', 5)
        self.assertEqual(set([('a', 6), ('aa', 6), ('aaa', 1), ('aab', 5)]),
                         set(self.trie.iterate('reg')))
        self.assertEqual(self.trie.size(), 4)

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

class OptimizingTableTest(unittest.TestCase):
    def test_table(self):
        ctable = pwd_guess.OptimizingCharacterTable('abcd', 2, 'ab', False)
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

    def test_table_upper(self):
        ctable = pwd_guess.OptimizingCharacterTable('abcdABCD:', 2, ':', True)
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
            min_len = 3,
            # trie_serializer_type = 'fuzzy',
            trie_implementation = 'disk'))
        pre.begin([('aaa', 1), ('caa', 2), ('aab', 5)])
        prefix, suffix, weight = pre.next_chunk()
        exp_prefix = ['', 'a', 'aa', 'aa', '', 'c', 'ca', 'aaa', 'aab', 'caa']
        exp_suffix = ['a', 'a', 'a', 'b', 'c', 'a', 'a', '\n', '\n', '\n']
        exp_weight = [6, 6, 1, 5, 2, 2, 2, 1, 5, 2]
        self.assertEqual(
            set(zip(exp_prefix, exp_suffix, exp_weight)),
            set(zip(prefix, suffix, weight)))

    def test_preprocess(self):
        pre = pwd_guess.HybridDiskPreprocessor(pwd_guess.ModelDefaults(
            min_len = 3,
            trie_serializer_type = 'fuzzy',
            trie_implementation = 'disk'))
        self.assertEqual([('aaa', 1), ('aab', 5), ('caa', 2)], list(
            pre.preprocess([('aaa', 1), ('caa', 2), ('aab', 5)])))

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
        self.assertEqual(pre.total_chunks(), 0)
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
        prefix, suffix, weight = p.next_chunk()
        expected_chunks = ['', 'p', 'pa', 'pas', 'pasw', 'pas', 'pass']
        expected_out = ['p', 'a', 's', 'w', '\n', 's', '\n']
        expected_weight = [5, 5, 5, 3, 3, 2, 2]
        self.assertEqual(
            set(zip(expected_chunks, expected_out, expected_weight)),
            set(zip(prefix, suffix, weight)))

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
        t = pwd_guess.Trainer(['pass'])
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

    def test_open_file(self):
        self.make_file('test.txt', open)
        pwd = pwd_guess.PwdList(self.fname)
        f = pwd.open_file()
        self.assertEqual(f.read(), self.fcontent)
        f.close()

    def test_open_gzip_file(self):
        self.make_file('test.txt.gz', gzip.open)
        pwd = pwd_guess.PwdList(self.fname)
        f = pwd.open_file()
        self.assertEqual(f.read(), self.fcontent)
        f.close()

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
            pwd_guess.PwdList.getFactory('tsv', pwd_guess.ModelDefaults(
                simulated_frequency_optimization = True)),
            pwd_guess.TsvSimulatedList)
        self.assertEqual(
            pwd_guess.PwdList.getFactory('tsv', pwd_guess.ModelDefaults(
                simulated_frequency_optimization = False)),
            pwd_guess.TsvList)
        self.assertEqual(
            pwd_guess.PwdList.getFactory('list', pwd_guess.ModelDefaults(
                simulated_frequency_optimization = False)),
            pwd_guess.PwdList)

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
        mock_model = Mock()
        mockable_return = [[distribution]]
        mock_model.predict = MagicMock(return_value = mockable_return)
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
        self.assertEqual([0.5, 0.5], guesser.conditional_probs(''))
        self.assertEqual([0.5, 0.5], guesser.conditional_probs('a'))
        self.assertEqual([0.5, 0.5], guesser.conditional_probs('aa'))
        self.assertEqual([0.5, 0.5], guesser.conditional_probs('aaa'))

    def test_relevel(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'a\n',
            lower_probability_threshold = 10**-1)
        guesser, ostream = self.make(config, [0.5, 0.5])
        self.assertEqual([0.0, 1.0], guesser.conditional_probs(''))
        self.assertEqual([0.0, 1.0], guesser.conditional_probs('a'))
        self.assertEqual([0.0, 1.0], guesser.conditional_probs('aa'))
        self.assertEqual([1.0, 0.0], guesser.conditional_probs('aaa'))

    def test_relevel_tri_alpha(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'ab\n',
            lower_probability_threshold = 10**-1)
        guesser, ostream = self.make(config, [0.5, 0.2, 0.3])
        self.assertEqual([0.0, .4, .6], guesser.conditional_probs(''))
        self.assertEqual([0.0, .4, .6], guesser.conditional_probs('a'))
        self.assertEqual([0.0, .4, .6], guesser.conditional_probs('aa'))
        self.assertEqual([0.0, .4, .6], guesser.conditional_probs('ab'))
        self.assertEqual([1.0, 0.0, 0.0], guesser.conditional_probs('aaa'))

    def test_do_guessing(self):
        config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'a\n',
            lower_probability_threshold = 10**-2,
            relevel_not_matching_passwords = False)
        model = self.mock_model(config, [0.5, 0.5])
        with tempfile.NamedTemporaryFile() as fp:
            pwd_guess.Guesser.do_guessing(model, config, fp.name)
            self.assertEqual("""	0.5
a	0.25
aa	0.125
aaa	0.0625
""", fp.read().decode('utf8'))

def mock_fork_starter(arguments):
    model = Mock()
    model.predict = MagicMock(return_value = [[[0.5, 0.25, 0.25]]])
    return pwd_guess.ParallelGuesser.fork_entry_point(model, arguments)

class ParallelGuesserTest(unittest.TestCase):
    mock_model = Mock()

    def setUp(self):
        self.mock_model.predict = MagicMock(return_value = [[[
            0.5, 0.25, 0.25]]])
        self.config = pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3, char_bag = 'ab\n',
            fork_length = 2)
        self.mock_output = io.StringIO()
        self.archfile = tempfile.NamedTemporaryFile()
        self.weightfile = tempfile.NamedTemporaryFile()
        self.serializer = Mock()
        self.serializer.archfile = self.archfile.name
        self.serializer.weightfile = self.weightfile.name
        self.serializer.load_model = MagicMock(return_value = self.mock_model)

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

    def test_get_argument_dict(self):
        parallel_guesser = pwd_guess.ParallelGuesser(
            self.serializer, self.config, self.mock_output)
        test_config = self.config.as_dict()
        prepared = parallel_guesser.prepare_argument_dict(['aa', 0.125])
        self.assertEqual({
            'config' : test_config,
            'serializer' : [self.archfile.name, self.weightfile.name],
            'node' : ['aa', 0.125],
            'ofile' : prepared['ofile']
        }, prepared)
        self.assertTrue(type(prepared['ofile']) == str)

    def test_forking(self):
        parallel_guesser = pwd_guess.ParallelGuesser(
            self.serializer, self.config, self.mock_output)
        parallel_guesser.fork_starter = mock_fork_starter
        parallel_guesser.guess()
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(io.StringIO(self.mock_output.getvalue()),
                               delimiter = '\t')]
        sort_freq = sorted(pwd_freq, key = lambda x: x[0])
        self.assertEqual(8, parallel_guesser.generated)
        self.assertEqual([('aaa', .125),
                          ('aab', .125),
                          ('aba', .125),
                          ('abb', .125),
                          ('baa', .125),
                          ('bab', .125),
                          ('bba', .125),
                          ('bbb', .125)], sort_freq)

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
        preprocessor = pwd_guess.BasePreprocessor.fromConfig(real_config)
        preprocessor.begin(pwd_guess.read_passwords(
            self.input_file.name, 'tsv', real_config))
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

if __name__ == '__main__':
    unittest.main()
