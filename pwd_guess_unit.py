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
import contextlib
import csv

import pwd_guess

class TrainerTest(unittest.TestCase):
    def test_training_set(self):
        t = pwd_guess.Trainer(['pass'], pwd_guess.ModelDefaults(max_len = 40))
        prefix, suffix = t.next_train_chunk()
        self.assertEqual(([x + ('\n' * (40 - len(x)))
                           for x in ['', 'p', 'pa', 'pas', 'pass']],
                          ['p', 'a', 's', 's', '\n']),
                         (list(prefix), list(suffix)))
        self.assertTrue(all(map(lambda x: len(x) == 40, list(prefix))))

    def test_accuracy(self):
        t = pwd_guess.Trainer(['pass'], pwd_guess.ModelDefaults(max_len = 5))
        mock_model = Mock()
        mock_model.train_on_batch = MagicMock(return_value = (0.5, 0.5))
        mock_model.test_on_batch = MagicMock(return_value = (0.5, 0.5))
        t.model = mock_model
        self.assertEqual(0.5, t.train_model_generation())

    def test_train_model(self):
        t = pwd_guess.Trainer(['pass'], pwd_guess.ModelDefaults(
            max_len = 5, generations = 20))
        mock_model = Mock()
        mock_model.train_on_batch = MagicMock(return_value = (0.5, 0.5))
        mock_model.test_on_batch = MagicMock(return_value = (0.5, 0.5))
        t.model = mock_model
        t.train_model(pwd_guess.ModelSerializer())
        self.assertEqual(t.generation, 2)

    def test_training_set_small(self):
        t = pwd_guess.Trainer(
            ['aaa'], pwd_guess.ModelDefaults(max_len = 3, min_len = 3))
        prefix, suffix = t.next_train_chunk()
        self.assertEqual(([x + ('\n' * (3 - len(x)))
                           for x in ['', 'a', 'aa', 'aaa']],
                          ['a', 'a', 'a', '\n']),
                         (list(prefix), list(suffix)))
        self.assertTrue(all(map(lambda x: len(x) == 3, list(prefix))))

    def test_char_table_no_error(self):
        t = pwd_guess.Trainer(['pass'])
        self.assertNotEqual(None, t.ctable)
        t.ctable.encode('1234' + ('\n' * 36), 40)

    def test_output_as_np(self):
        t = pwd_guess.Trainer(['pass'])
        t.next_train_set_as_np()

    def test_build_model(self):
        t = pwd_guess.Trainer(['pass'])
        t.build_model()
        self.assertNotEqual(None, t.model)

    def test_train_set_np_two(self):
        t = pwd_guess.Trainer(['pass', 'word'])
        t.next_train_set_as_np()

    def test_train_set_np_digits(self):
        t = pwd_guess.Trainer(['pass', '1235', '<>;p[003]', '$$$$$$ '],
                              pwd_guess.ModelDefaults(
                                  training_chunk = 1, visualize_errors = False))
        m = t.train(pwd_guess.ModelSerializer())
        self.assertEqual(4, t.chunk)

    def test_test_set(self):
        t = pwd_guess.Trainer([], pwd_guess.ModelDefaults(
            train_test_ratio = 10))
        a = np.zeros((10, 1, 1), dtype = np.bool)
        b = np.zeros((10, 1, 1), dtype = np.bool)
        x_t, x_v, y_t, y_v = t.test_set(a, b)
        self.assertEqual(9, len(x_t))
        self.assertEqual(1, len(x_v))
        self.assertEqual(9, len(y_t))
        self.assertEqual(1, len(y_v))

    def test_test_set_small(self):
        t = pwd_guess.Trainer([], pwd_guess.ModelDefaults(
            train_test_ratio = 10))
        a = np.zeros((5, 1, 1), dtype = np.bool)
        b = np.zeros((5, 1, 1), dtype = np.bool)
        x_t, x_v, y_t, y_v = t.test_set(a, b)
        self.assertEqual(4, len(x_t))
        self.assertEqual(1, len(x_v))
        self.assertEqual(4, len(y_t))
        self.assertEqual(1, len(y_v))


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
        self.assertEqual(['pass ', 'word'], pwd.as_list())

    def test_tsv(self):
        self.fcontent = 'pass \t1\tR\nword\t1\tR\n'
        self.make_file('test.tsv', open)
        pwd = pwd_guess.TsvList(self.fname)
        self.assertEqual(['pass ', 'word'], pwd.as_list())

    def test_tsv_multiplier(self):
        self.fcontent = 'pass \t2\tR\nword\t1\tR\n'
        self.make_file('test.tsv', open)
        pwd = pwd_guess.TsvList(self.fname)
        self.assertEqual(['pass ', 'pass ', 'word'], pwd.as_list())

    def test_tsv_quote_char(self):
        self.fcontent = 'pass"\t1\tR\nword\t1\tR\n'
        self.make_file('test.tsv', open)
        pwd = pwd_guess.TsvList(self.fname)
        self.assertEqual(['pass"', 'word'], pwd.as_list())

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
        self.assertEqual(['pass'], list(f.filter(['asdf£jfj', 'pass'])))

    def test_filter_small(self):
        f = pwd_guess.Filterer(pwd_guess.ModelDefaults(
            min_len = 3, max_len = 3))
        self.assertEqual(['aaa'], list(f.filter(['aaa'])))

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
        self.assertEqual("""aaa	0.0625
aa	0.125
a	0.25
	0.5
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
            self.assertEqual("""aaa	0.0625
aa	0.125
a	0.25
	0.5
""", fp.read().decode('utf8'))

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

class EndToEndTest(unittest.TestCase):

    skewed_dict = ['abab', 'abbbb', 'aaaa', 'aaab']
    probs = [0.1, 0.4, 0.2, 0.3]


    def skewed(self):
        return self.skewed_dict[
            np.random.choice(len(self.skewed_dict), 1, p = self.probs)[0]]

    def make_dist(self, line_count, dist):
        fun = self.skewed if dist == 'skewed' else lambda: 'aaa'
        for _ in range(line_count):
            self.input_file.write('%s\n' % fun())

    def setUp(self):
        self.config_file, self.output_file, self.input_file = (
            tempfile.NamedTemporaryFile(mode = 'w'),
            tempfile.NamedTemporaryFile(mode = 'r'),
            tempfile.NamedTemporaryFile(mode = 'w'))

    def test_skewed(self):
        json.dump({
            "chunk_print_interval" : 100,
            "training_chunk" : 64,
            "layers" : 3,
            "hidden_size" : 256,
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
            "hidden_size" : 256,
            "generations" : 20,
            "min_len" : 3,
            "max_len" : 5,
            "char_bag" : "ab\n"
        }, self.config_file)
        self.make_dist(10000, 'constant')
        self.config_file.flush()
        self.input_file.flush()
        pwd_guess.main(vars(pwd_guess.make_parser().parse_args([
            '--pwd-file', self.input_file.name,
            '--config', self.config_file.name,
            '--enumerate-ofile', self.output_file.name,
            '--log-level', 'error'
        ])))
        pwd_freq = [(row[0], float(row[1])) for row in
                    csv.reader(self.output_file, delimiter = '\t')]
        sort_freq = list(
            map(lambda x: x[0],
                sorted(pwd_freq, key = lambda x: x[1], reverse = True)))
        self.assertEqual(['aaa'], sort_freq[:1])

if __name__ == '__main__':
    unittest.main()
