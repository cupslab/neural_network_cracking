import unittest
import tempfile
import io

import strategy_calculator as sc

class ReadfileTests(unittest.TestCase):
    def test_read_prob(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as tf:
            tf.write('asdf\t0.2\njjj\t.1\n')
            tf.flush()
            self.assertEqual([('asdf', .2), ('jjj', .1)],
                             list(sc.read_probability_file(tf.name)))

    def test_read_prob_error(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as tf:
            tf.write('asdf\t0.05\njjj\t.1\n')
            tf.flush()
            try:
                a = list(sc.read_probability_file(tf.name))
                self.fail()
            except AssertionError as e:
                self.assertEqual('Out of order probability', str(e))

    def test_read_pass(self):
        with tempfile.NamedTemporaryFile(mode = 'w') as tf:
            tf.write('asdf\njjj')
            tf.flush()
            self.assertEqual(['asdf', 'jjj'],
                             list(sc.read_password_file(tf.name)))

class NaiveStrategyTest(unittest.TestCase):
    def test_calculate_index_map(self):
        a = sc.NaiveStrategy([('asdf', .2), ('jjj', .1)],
                             ['asdf', 'jjj', 'pwd'])
        self.assertEqual({
            'pwd' : [2], 'jjj' : [1], 'asdf' : [0]
        }, a.calculate_index_map())

    def test_calculate_index_map_many(self):
        a = sc.NaiveStrategy([('asdf', .2), ('jjj', .1)],
                             ['asdf', 'jjj', 'pwd', 'pwd'])
        self.assertEqual({
            'pwd' : [2, 3], 'jjj' : [1], 'asdf' : [0]
        }, a.calculate_index_map())

    def test_calculate_hash_nums(self):
        a = sc.NaiveStrategy([('asdf', .2), ('jjj', .1),
                              ('pwd', .05), ('test', .01)],
                             ['jjj', 'pwd', 'ppppppp', 'asdf'])
        self.assertEqual([5, 8, -1, 4], a.calculate_hash_nums())

    def test_calculate_hash_nums_many(self):
        a = sc.NaiveStrategy([('asdf', .2), ('jjj', .1),
                              ('pwd', .05), ('test', .01)],
                             ['jjj', 'pwd', 'ppppppp', 'asdf', 'asdf'])
        self.assertEqual([6, 9, -1, 4, 5], a.calculate_hash_nums())

    def test_calculate_hash_output(self):
        a = sc.NaiveStrategy([('asdf', .2), ('jjj', .1),
                              ('pwd', .05), ('test', .01)],
                             ['jjj', 'pwd', 'ppppppp', 'asdf', 'asdf'])
        mock_io = io.StringIO()
        a.output(mock_io)
        self.assertEqual('\n'.join(map(str, [4, 5, 6, 9, -1])) + '\n',
                         mock_io.getvalue())

    def test_calculate_hash_no_cracks(self):
        a = sc.NaiveStrategy([('asdf', .2), ('jjj', .1),
                              ('pwd', .05), ('test', .01)], ['uuuu', 'iiiii'])
        mock_io = io.StringIO()
        a.output(mock_io)
        self.assertEqual('\n'.join(map(str, [-1, -1])) + '\n',
                         mock_io.getvalue())

    def test_calculate_all_cracks(self):
        a = sc.NaiveStrategy([('asdf', .2), ('jjj', .1),
                              ('pwd', .05), ('test', .01), ('ppppppp', .001)],
                             ['jjj', 'pwd', 'ppppppp', 'asdf', 'asdf'])
        mock_io = io.StringIO()
        a.output(mock_io)
        self.assertEqual('\n'.join(map(str, [4, 5, 6, 9, 12])) + '\n',
                         mock_io.getvalue())
