import unittest
from unittest.mock import Mock, MagicMock
import string
import tempfile
import io

import numpy as np

import pwd_guess as pg
import markov_model as mm

class MarkovModelTest(unittest.TestCase):
    def test_train_one_pwd_no_smoothing(self):
        config = Mock()
        config.char_bag = string.ascii_lowercase + pg.PASSWORD_END
        m = mm.MarkovModel(config, smoothing='none', order=2)
        m.train([('pass', 1)])
        self.assertAlmostEqual(m.probability_next_char('p', 'a'), 1)
        self.assertAlmostEqual(m.probability_next_char('pa', 's'), 1)
        self.assertAlmostEqual(
            m.probability_next_char('pass', pg.PASSWORD_END), .5)
        self.assertAlmostEqual(m.probability_next_char('pas', 's'), .5)
        self.assertAlmostEqual(m.probability_next_char('', 'p'), 1)

        self.assertAlmostEqual(m.probability_next_char('pas', 'k'), 0)
        self.assertAlmostEqual(m.probability_next_char('', 'j'), 0)


    def test_train_two_pwd_no_smoothing(self):
        config = Mock()
        config.char_bag = string.ascii_lowercase + pg.PASSWORD_END
        m = mm.MarkovModel(config, smoothing='none', order=2)
        m.train([('pass', 1), ('past', 1)])
        self.assertAlmostEqual(m.probability_next_char('', 'p'), 1)
        self.assertAlmostEqual(m.probability_next_char('p', 'a'), 1)
        self.assertAlmostEqual(m.probability_next_char('pa', 's'), 1)
        self.assertAlmostEqual(
            m.probability_next_char('pass', pg.PASSWORD_END), 1/3.)
        self.assertAlmostEqual(m.probability_next_char('pas', 's'), 1/3.)
        self.assertAlmostEqual(m.probability_next_char('pas', 't'), 1/3.)

        self.assertAlmostEqual(m.probability_next_char('pas', 'k'), 0)
        self.assertAlmostEqual(m.probability_next_char('', 'j'), 0)


    def test_train_pwd_long(self):
        config = Mock()
        config.char_bag = string.ascii_lowercase + pg.PASSWORD_END
        m = mm.MarkovModel(config, smoothing='none', order=4)
        m.train([('pa', 1)])
        self.assertEqual(m.freq_dict, {
            'p' : 1, 'pa' : 1, 'pa' + pg.PASSWORD_END : 1
        })

    def test_train_high_order_no_smoothing(self):
        config = Mock()
        config.char_bag = string.ascii_lowercase + pg.PASSWORD_END
        m = mm.MarkovModel(config, smoothing='none', order=3)
        m.train([('pass', 1), ('past', 1), ('ashen', 1)])
        self.assertAlmostEqual(m.probability_next_char('', 'p'), 2./3.)
        self.assertAlmostEqual(m.probability_next_char('p', 'a'), 1)
        self.assertAlmostEqual(m.probability_next_char('pa', 's'), 1)
        self.assertAlmostEqual(
            m.probability_next_char('pass', pg.PASSWORD_END), 1)
        self.assertAlmostEqual(m.probability_next_char('pas', 's'), 1./3.)
        self.assertAlmostEqual(m.probability_next_char('pas', 't'), 1./3.)
        self.assertAlmostEqual(m.probability_next_char('as', 'h'), 1./3.)
        self.assertAlmostEqual(m.probability_next_char('as', 't'), 1./3.)

        self.assertAlmostEqual(m.probability_next_char('pas', 'k'), 0)
        self.assertAlmostEqual(m.probability_next_char('', 'j'), 0)


    def test_save_load_model(self):
        config = Mock()
        config.char_bag = string.ascii_lowercase + pg.PASSWORD_END
        m = mm.MarkovModel(config, smoothing='none', order=2)
        m.train([('pass', 1), ('past', 1), ('ashen', 1)])
        self.assertAlmostEqual(m.probability_next_char('', 'p'), 2./3.)

        with tempfile.NamedTemporaryFile('w') as tempf:
            m.saveModel(tempf.name)
            new_model = mm.MarkovModel.fromModelFile(
                tempf.name, config, smoothing='none', order=2)

        self.assertAlmostEqual(new_model.probability_next_char('', 'p'), 2./3.)

    def test_predict(self):
        config = Mock()
        config.char_bag = pg.PASSWORD_END + 'aehnpst'
        m = mm.MarkovModel(config, smoothing='none', order=2)
        m.train([('pass', 1), ('past', 1), ('ashen', 1)])
        answer = np.zeros((len(config.char_bag), ), dtype=np.float64)
        m.predict('pa', answer)
        np.testing.assert_array_equal(answer, np.array([
            0, 0, 0, 0, 0, 0, 1, 0
        ]))

class MarkovGuesserTest(unittest.TestCase):
    def test_build(self):
        config = pg.ModelDefaults(char_bag = pg.PASSWORD_END + 'aehnpst',
                                  guesser_class = 'markov_model')
        pg.GuesserBuilder.other_class_builders[
            'markov_model'] = mm.MarkovGuesser
        model = mm.MarkovModel(config, smoothing='none', order=2)
        model.train([('pass', 1), ('past', 1), ('ashen', 1)])
        guesser_builder = pg.GuesserBuilder(config)
        guesser_builder.add_model(model)
        ostream = io.StringIO()
        guesser_builder.add_stream(ostream)
        guesser = guesser_builder.build()
        self.assertEqual(type(guesser), mm.MarkovGuesser)
        np.testing.assert_array_almost_equal(
            guesser.conditional_probs_many(['pa']), np.array([[[
                0, 0, 0, 0, 0, 0, 1, 0
            ]]], dtype=np.float64))


class AdditiveSmoothingTest(unittest.TestCase):
    def test_predict(self):
        config = Mock()
        config.char_bag = 'abc'
        config.additive_smoothing_amount = 1
        sm = mm.AdditiveSmoothingSmoother({
            'a' : 1,
            'b' : 2
        }, config)
        answer = np.zeros((3, ), dtype=np.float64)
        sm.predict('', answer)
        np.testing.assert_array_almost_equal(
            answer, np.array([1/3., 1/2., 1/6.], dtype=np.float64))

class BackoffMarkovModelTest(unittest.TestCase):
    def test_train_one(self):
        config = Mock()
        config.char_bag = (
            string.ascii_lowercase + pg.PASSWORD_END)
        m = mm.BackoffMarkovModel(config, order=2)
        m.train([('pass', 1)])
        self.assertEqual(set(m.freq_dict.items()), set([
            (mm.PASSWORD_START, 1), ('p', 1), ('a', 1), ('s', 2),
            (pg.PASSWORD_END, 1), (mm.PASSWORD_START + 'p', 1),
            ('pa', 1), ('as', 1), ('ss', 1), ('s' + pg.PASSWORD_END, 1)
        ]))

    def test_train_two(self):
        config = Mock()
        config.char_bag = (
            string.ascii_lowercase + pg.PASSWORD_END)
        m = mm.BackoffMarkovModel(config, order=2)
        m.train([('pass', 1), ('task', 1)])
        self.assertEqual(set(m.freq_dict.items()), set([
            (mm.PASSWORD_START, 2),
            (pg.PASSWORD_END, 2),
            ('p', 1),
            ('a', 2),
            ('s', 3),
            ('k', 1),
            ('t', 1),
            (mm.PASSWORD_START + 'p', 1),
            ('pa', 1),
            ('as', 2),
            ('ss', 1),
            ('s' + pg.PASSWORD_END, 1),
            (mm.PASSWORD_START + 't', 1),
            ('ta', 1),
            ('sk', 1),
            ('k' + pg.PASSWORD_END, 1)
        ]))

    def test_predict_short_context(self):
        config = Mock()
        config.char_bag = ('abc' + pg.PASSWORD_END)
        config.backoff_smoothing_threshold = 0
        config.additive_smoothing_amount = 0
        m = mm.BackoffMarkovModel(config, order=2)
        m.train([('abc', 1)])
        answer = np.zeros((len(config.char_bag), ), dtype=np.float64)
        m.predict('ab', answer)
        np.testing.assert_array_almost_equal(
            answer, np.array([0., 0., 0., 1.], dtype=np.float64))
        answer.fill(0)
        m.predict('ba', answer)
        np.testing.assert_array_almost_equal(
            answer, np.array([0., 0., 1., 0.], dtype=np.float64))

    def test_predict_longer_context(self):
        config = Mock()
        config.char_bag = ('abc' + pg.PASSWORD_END)
        config.backoff_smoothing_threshold = 0
        config.additive_smoothing_amount = 0
        m = mm.BackoffMarkovModel(config, order=3)
        m.train([('abc', 1), ('aaa', 1)])
        answer = np.zeros((len(config.char_bag), ), dtype=np.float64)
        m.predict('ab', answer)
        np.testing.assert_array_almost_equal(
            answer, np.array([0., 0., 0., 1.], dtype=np.float64))
        answer.fill(0)
        m.predict('ba', answer)
        np.testing.assert_array_almost_equal(
            answer, np.array([.25, .5, .25, 0.], dtype=np.float64))


if __name__=='__main__':
    unittest.main()
