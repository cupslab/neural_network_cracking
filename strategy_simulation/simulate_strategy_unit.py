import unittest
from unittest.mock import MagicMock, Mock
import io
import os

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
        tabulator.record(True)
        tabulator.record(False)
        tabulator.record(False)
        tabulator.record(True)
        ostream = io.StringIO()
        tabulator.output(ostream)
        self.assertEqual(os.linesep.join(['1\t3', '1\t6']) + os.linesep,
                         ostream.getvalue())

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
        tabulator.output(ostream)
        self.assertEqual(os.linesep.join(['1\t3', '1\t6', '2\t2']) + os.linesep,
                         ostream.getvalue())
