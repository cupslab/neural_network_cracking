import sys
import argparse
import random
import collections
import csv
import os

class Strategy(object):
    def __init__(self, num_guessing_set):
        self.num_guessing_set = num_guessing_set
        self.guessed_idxes = set()

    def make_state(self, pwd_prior_iter):
        raise NotImplementedError()

    def next_action(self):
        raise NotImplementedError()

    def store_result(self, outcome):
        raise NotImplementedError()

class TreeStrat(Strategy):
    def make_state(self, pwd_prior_iter):
        self.pwd_priors = list(pwd_prior_iter)
        self.guessed_idxes = set()
        self.pwd_guessing_state = [0] * self.num_guessing_set
        self.pwd_guessing_prob_total = [1] * self.num_guessing_set

    def next_prob_at(self, idx):
        pwd_idx = self.pwd_guessing_state[idx]
        if pwd_idx < len(self.pwd_priors):
            return ((self.pwd_priors[pwd_idx][1] /
                     self.pwd_guessing_prob_total[idx]),
                    self.pwd_priors[pwd_idx][0])
        else:
            return -1, ''

    def next_action(self):
        best_prob = 0
        best_pwd = ''
        best_idx = -1
        for i in range(self.num_guessing_set):
            if i in self.guessed_idxes:
                continue
            prob, pwd = self.next_prob_at(i)
            if best_prob < prob:
                best_prob = prob
                best_pwd = pwd
                best_idx = i
        if best_idx == -1:
            return False
        self.cur_idx = best_idx
        self.cur_pwd = best_pwd
        self.cur_prob = best_prob
        return self.cur_pwd, self.cur_idx

    def store_result(self, outcome):
        if outcome:
            self.guessed_idxes.add(self.cur_idx)
        else:
            self.pwd_guessing_prob_total[self.cur_idx] -= self.cur_prob
            self.pwd_guessing_state[self.cur_idx] += 1

class NaiveStrat(Strategy):
    def make_state(self, pwd_prior_iter):
        self.current_idx = 0
        self.pwd_prior_iter = iter(pwd_prior_iter)
        self.cur_prob = 1
        self.next_pwd()
        self.guessed_idxes = set()

    def next_action(self):
        if not self.cur_password:
            return False
        return self.cur_password, self.current_idx

    def next_pwd(self):
        prev_prob = self.cur_prob
        self.current_idx = 0
        try:
            self.cur_password, self.cur_prob = self.pwd_prior_iter.__next__()
        except StopIteration as e:
            self.cur_password, self.cur_prob = False, 0
        assert prev_prob >= self.cur_prob, 'Password file must be sorted'

    def inc_idx(self):
        self.current_idx += 1
        if self.current_idx >= self.num_guessing_set:
            self.next_pwd()

    def good_idx(self):
        return self.current_idx not in self.guessed_idxes

    def next_idx(self):
        self.inc_idx()
        while not self.good_idx() and self.cur_password:
            self.inc_idx()

    def store_result(self, outcome):
        if outcome:
            self.guessed_idxes.add(self.current_idx)
        self.next_idx()

class Tabulator(object):
    def __init__(self):
        self.runs = [[]]
        self.run_num = 0
        self.hash_num = 0
        self.cracked_num = 0

    def reset(self):
        self.run_num += 1
        self.hash_num = 0
        self.cracked_num = 0
        self.runs.append([])

    def record(self, outcome):
        self.cracked_num += 1
        if outcome:
            self.runs[self.run_num].append(self.cracked_num)

    def output(self, ofile):
        writer = csv.writer(ofile, delimiter = '\t', quotechar = None,
                            lineterminator = os.linesep)
        average_accum = 0
        for run_idx in range(len(self.runs)):
            hashes = self.runs[run_idx]
            for ahash_num in hashes:
                average_accum += 1
                writer.writerow([run_idx + 1, ahash_num])
        sys.stdout.write(str(average_accum / len(self.runs)) + '\n')

class Simulator(object):
    def __init__(self, strategy_factory, test_pwds, pwd_priors, tabulator):
        self.strategy = strategy_factory(len(test_pwds))
        self.strategy.make_state(pwd_priors)
        self.test_pwds = test_pwds
        self.tabulator = tabulator

    def run_step(self):
        action = self.strategy.next_action()
        if not action:
            return True
        pwd, idx = action
        outcome = (self.test_pwds[idx] == pwd)
        self.tabulator.record(outcome)
        self.strategy.store_result(outcome)

    def run(self, nsteps):
        for _ in range(nsteps):
            if self.run_step():
                break

strategy_map = {
    'naive': NaiveStrat,
    'tree' : TreeStrat
}

def read_prob_file(pwd_file):
    with open(pwd_file, 'r') as pwd_file_stream:
        for row in csv.reader(
                pwd_file_stream, delimiter = '\t', quotechar = None):
            yield (row[0], float(row[1]))

def read_test_pwds(pwd_file):
    with open(pwd_file, 'r') as test_pwd_file:
        return [line.strip(os.linesep) for line in test_pwd_file]

def main(args):
    test_pwds = read_test_pwds(args.test_file)
    tabulator = Tabulator()
    probs = list(read_prob_file(args.prob_file))
    for _ in range(args.num_runs):
        random.shuffle(test_pwds)
        sim = Simulator(strategy_map[args.strategy], test_pwds,
                        probs, tabulator)
        sim.run(args.num)
        tabulator.reset()
    with open(args.ofile, 'w') as ostream:
        tabulator.output(ostream)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('test_file', help='Input file with test passwords. ')
    parser.add_argument('prob_file', help='Input file with probabilities. ')
    parser.add_argument('strategy', choices = sorted(strategy_map.keys()))
    parser.add_argument('ofile', help='Output file')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--num-runs', type=int, default=10)
    main(parser.parse_args())
