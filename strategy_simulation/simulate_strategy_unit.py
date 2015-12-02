import unittest
from unittest.mock import MagicMock, Mock
import io
import os
import csv

import simulate_strategy as sim

class TestNaiveStrat(unittest.TestCase):
    def test_simple(self):
        strat = sim.NaiveStrat(3)
        pwds = [('a', .2), ('b', .1)]
        strat.make_state(pwds)
        self.assertEqual(strat.next_action(), ('a', 0))
        strat.store_result(False)
        self.assertEqual(strat.next_action(), ('a', 1))
        strat.store_result(False)
        self.assertEqual(strat.next_action(), ('a', 2))
        strat.store_result(False)
        self.assertEqual(strat.next_action(), ('b', 0))

    def test_solve(self):
        strat = sim.NaiveStrat(3)
        pwds = [('a', .2), ('b', .1)]
        strat.make_state(pwds)
        self.assertEqual(strat.next_action(), ('a', 0))
        strat.store_result(True)
        self.assertEqual(strat.next_action(), ('a', 1))
        strat.store_result(False)
        self.assertEqual(strat.next_action(), ('a', 2))
        strat.store_result(False)
        self.assertEqual(strat.next_action(), ('b', 1))

    def test_end(self):
        strat = sim.NaiveStrat(3)
        pwds = [('a', .2), ('b', .1)]
        strat.make_state(pwds)
        for _ in range(6):
            self.assertTrue(not not strat.next_action())
            strat.store_result(False)
        self.assertFalse(strat.next_action())

class TestTreeStrat(unittest.TestCase):
    def test_tree(self):
        strat = sim.TreeStrat(2)
        strat.make_state([('a', .2), ('b', .19)])
        self.assertAlmostEqual(strat.next_prob_at(0)[0], 0.2)
        self.assertAlmostEqual(strat.next_prob_at(1)[0], 0.2)
        self.assertEqual(strat.next_action(), ('a', 0))
        strat.store_result(False)
        self.assertAlmostEqual(strat.next_prob_at(0)[0], (0.19/0.8))
        self.assertAlmostEqual(strat.next_prob_at(1)[0], 0.2)
        self.assertEqual(strat.next_action(), ('b', 0))

    def test_tree_end(self):
        strat = sim.TreeStrat(2)
        strat.make_state([('a', .2), ('b', .19)])
        for _ in range(4):
            self.assertTrue(not not strat.next_action())
            strat.store_result(False)
        self.assertFalse(strat.next_action())

    def test_real_end(self):
        strat = sim.TreeStrat(2)
        strat.make_state([('654321', 0.00247049666073),
                          ('987654321', 0.000802497539062)])
        self.assertEqual(strat.next_action(), ('654321', 0))
        strat.store_result(False)
        self.assertEqual(strat.next_action(), ('654321', 1))
        strat.store_result(False)

    def test_real_start(self):
        strat = sim.TreeStrat(2)
        strat.make_state([('654321', 0.00247049666073),
                          ('987654321', 0.000802497539062)])
        self.assertEqual(strat.next_action(), ('654321', 0))
        strat.store_result(False)
        self.assertEqual(strat.next_action(), ('654321', 1))
        strat.store_result(False)

    def test_real_data(self):
        strat = sim.TreeStrat(2)
        with open('test_data/sorted.guess_file.tsv', 'r') as sg:
            guesses = [(row[0], float(row[1])) for row in csv.reader(
                sg, delimiter = '\t', quotechar = None)]
        strat.make_state(guesses)
        for i in range(273):
            pwd, idx = strat.next_action()
            self.assertEqual(guesses[int(i/2)][0], pwd)
            self.assertEqual(i % 2, idx)
            strat.store_result(False)
        self.assertAlmostEqual(strat.next_prob_at(0)[0], 3.44507E-05)
        pwd, idx = strat.next_action()
        self.assertEqual(pwd, 'angelito')
        self.assertEqual(idx, 0)

class TestBayesStrat(unittest.TestCase):
    def test_tree(self):
        strat = sim.BayesStrat(2)
        strat.make_state([('a', .2), ('b', .19)])
        self.assertAlmostEqual(strat.next_prob_at(0)[0], 0.2)
        self.assertAlmostEqual(strat.next_prob_at(1)[0], 0.2)
        self.assertEqual(strat.next_action(), ('a', 0))
        strat.store_result(False)
        self.assertAlmostEqual(strat.next_prob_at(0)[0], 0.19/0.8)
        self.assertAlmostEqual(strat.next_prob_at(1)[0], 0.2)
        self.assertEqual(strat.next_action(), ('b', 0))

    def test_positive(self):
        strat = sim.BayesStrat(2)
        strat.make_state([('a', .2), ('b', .1)])
        strat.original_weight = 5
        self.assertAlmostEqual(strat.next_prob_at(0)[0], 0.2)
        self.assertAlmostEqual(strat.next_prob_at(1)[0], 0.2)
        self.assertEqual(strat.next_action(), ('a', 0))
        strat.store_result(True)
        self.assertAlmostEqual(strat.next_prob_at(1)[0], (1/3))
        self.assertEqual(strat.next_action(), ('a', 1))

class TestSimulator(unittest.TestCase):
    def test_run_one(self):
        answer = Mock()
        answer.next_action = MagicMock(return_value = ('a', 0))
        mock_tabulator = Mock()
        mock_tabulator.record = MagicMock()
        simulator = sim.Simulator(
            lambda _: answer, ['a', 'b', 'c'], [], mock_tabulator)
        simulator.run(1)
        mock_tabulator.record.assert_called_with(True)

    def test_run_two(self):
        answer = Mock()
        answer.next_action = MagicMock(return_value = ('a', 0))
        mock_tabulator = Mock()
        mock_tabulator.record = MagicMock()
        simulator = sim.Simulator(
            lambda _: answer, ['a', 'b', 'c'], [], mock_tabulator)
        simulator.run(5)
        self.assertTrue(
            mock_tabulator.record.call_args_list == [((True,),)] * 5)

    def test_run_break(self):
        answer = Mock()
        answer.next_action = MagicMock(return_value = False)
        mock_tabulator = Mock()
        mock_tabulator.record = MagicMock()
        simulator = sim.Simulator(
            lambda _: answer, ['a', 'b', 'c'], [], mock_tabulator)
        simulator.run(10)
        self.assertTrue(
            mock_tabulator.record.call_args_list == [])

    def test_run_diff(self):
        answer = Mock()
        answer.next_action = MagicMock(return_value = ('a', 0))
        mock_tabulator = Mock()
        mock_tabulator.record = MagicMock()
        simulator = sim.Simulator(
            lambda _: answer, ['a', 'b', 'c'], [], mock_tabulator)
        simulator.run_step()
        mock_tabulator.record.assert_called_once_with(True)
        answer.next_action = MagicMock(return_value = ('c', 1))
        mock_tabulator.record.reset_mock()
        simulator.run_step()
        mock_tabulator.record.assert_called_once_with(False)

class TestTabulator(unittest.TestCase):
    def test_record(self):
        tabulator = sim.Tabulator()
        tabulator.record(False)
        tabulator.record(False)
        tabulator.record(True, 'asdf')
        tabulator.record(False)
        tabulator.record(False)
        tabulator.record(True)
        tabulator.reset()
        ostream = io.StringIO()
        summary_ofile = io.StringIO()
        tabulator.output(ostream, summary_ofile)
        self.assertEqual(os.linesep.join(['1\t3\tasdf', '1\t6\t']) + os.linesep,
                         ostream.getvalue())
        self.assertEqual('2.0\n', summary_ofile.getvalue())

    def test_record_many(self):
        tabulator = sim.Tabulator()
        tabulator.record(False)
        tabulator.record(False)
        tabulator.record(True)
        tabulator.record(False)
        tabulator.record(False)
        tabulator.record(True)
        tabulator.reset()
        tabulator.record(False)
        tabulator.record(True)
        tabulator.record(False)
        ostream = io.StringIO()
        summary_ofile = io.StringIO()
        tabulator.output(ostream, summary_ofile)
        self.assertEqual(os.linesep.join(
            ['1\t3\t', '1\t6\t', '2\t2\t']) + os.linesep,
                         ostream.getvalue())
        self.assertEqual('3.0\n', summary_ofile.getvalue())
