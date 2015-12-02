import unittest
from unittest.mock import MagicMock, Mock
import tempfile
import shutil
import os.path
import gzip

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

    def test_char_table_no_error(self):
        t = pwd_guess.Trainer(['pass'])
        self.assertNotEqual(None, t.ctable)
        t.ctable.encode('1234' + ('\n' * 36), 40)

    def test_output_as_np(self):
        t = pwd_guess.Trainer(['pass'])
        t.next_train_set_as_np()

    def test_verification_set(self):
        t = pwd_guess.Trainer(['pass', 'word'],
                              pwd_guess.ModelDefaults(training_chunk = 1,
                                                      train_test_split = 0.2))
        x_all, y_all = t.next_train_set_as_np()
        s = t.verification_set(x_all, y_all)
        t_x, v_x, t_y, v_y = s
        self.assertEqual(4, len(t_x))
        self.assertEqual(1, len(v_x))
        self.assertEqual(4, len(t_y))
        self.assertEqual(1, len(v_y))
        x_all, y_all = t.next_train_set_as_np()
        s = t.verification_set(x_all, y_all)
        t_x, v_x, t_y, v_y = s
        self.assertEqual(4, len(t_x))
        self.assertEqual(1, len(v_x))
        self.assertEqual(4, len(t_y))
        self.assertEqual(1, len(v_y))

    def test_build_model(self):
        t = pwd_guess.Trainer(['pass'])
        self.assertNotEqual(None, t.build_model())

    def test_train_set_np_two(self):
        t = pwd_guess.Trainer(['pass', 'word'])
        t.next_train_set_as_np()

    def test_train_set_np_digits(self):
        t = pwd_guess.Trainer(['pass', '1235', '<>;p[003]', '$$$$$$ '])
        t.next_train_set_as_np()

    def test_train_set_np_digits(self):
        t = pwd_guess.Trainer(['pass', '1235', '<>;p[003]', '$$$$$$ '],
                              pwd_guess.ModelDefaults(
                                  training_chunk = 1, visualize_errors = False))
        m = t.train()
        self.assertEqual(4, t.chunk)

class ModelDefaultsTest(unittest.TestCase):
    def test_get_default(self):
        m = pwd_guess.ModelDefaults()
        self.assertEqual(pwd_guess.ModelDefaults.hidden_size, m.hidden_size)

    def test_get_set(self):
        m = pwd_guess.ModelDefaults(hidden_size = 8)
        self.assertEqual(8, m.hidden_size)

    def test_get_set_dict(self):
        m = pwd_guess.ModelDefaults({'hidden_size' : 8})
        self.assertEqual(8, m.hidden_size)

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

# TODO: test for model training too short

class ModelSerializerTest(unittest.TestCase):
    def test_model_serializer(self):
        mock = Mock()
        write_value = '{1234}'
        mock.to_json = MagicMock(return_value = write_value)
        mock.save_weights = MagicMock()
        with tempfile.NamedTemporaryFile() as fp:
            with tempfile.NamedTemporaryFile() as tp:
                serializer = pwd_guess.ModelSerializer(fp.name, tp.name)
                serializer.save_model(mock)
                self.assertEqual(write_value, fp.read().decode('utf8'))

if __name__ == '__main__':
    unittest.main()
