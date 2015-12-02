import sys
import argparse
import random
import collections
import csv
import os
import heapq

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
        self.queue = []
        for i in range(self.num_guessing_set):
            pwd, prob = self.pwd_priors[0]
            heapq.heappush(self.queue, (1 - prob, prob, pwd, i))

    def next_prob_at(self, idx):
        pwd_idx = self.pwd_guessing_state[idx]
        if pwd_idx < len(self.pwd_priors):
            return ((self.pwd_priors[pwd_idx][1] /
                     self.pwd_guessing_prob_total[idx]),
                    self.pwd_priors[pwd_idx][0])
        else:
            return -1, ''

    def next_action(self):
        if len(self.queue) == 0:
            return False
        probinv, prob, pwd, idx = heapq.heappop(self.queue)
        self.cur_prob = prob
        self.cur_pwd = pwd
        self.cur_idx = idx
        return self.cur_pwd, self.cur_idx

    def store_result(self, outcome):
        if outcome:
            self.guessed_idxes.add(self.cur_idx)
        else:
            self.pwd_guessing_prob_total[self.cur_idx] -= self.cur_prob
            self.pwd_guessing_state[self.cur_idx] += 1
            prob, pwd = self.next_prob_at(self.cur_idx)
            if prob != -1:
                heapq.heappush(self.queue, (1 - prob, prob, pwd, self.cur_idx))

class BayesStrat(TreeStrat):
    def make_state(self, pwd_prior_iter):
        super().make_state(pwd_prior_iter)
        self.observed = 0
        self.posteriors = [0] * len(self.pwd_priors)
        self.original_priors = self.pwd_priors[:]
        self.original_weight = 100

    def successful_update_priors(self):
        pwd_idx = self.pwd_guessing_state[self.cur_idx]
        self.observed += 1
        self.posteriors[pwd_idx] += 1
        self.pwd_priors[pwd_idx] = (
            self.cur_pwd, ((self.original_priors[pwd_idx][1] *
                            self.original_weight + self.posteriors[pwd_idx]) /
                           (self.original_weight + self.observed)))

    def store_result(self, outcome):
        super().store_result(outcome)
        if outcome:
            self.successful_update_priors()

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

    def record(self, outcome, pwd = None):
        self.cracked_num += 1
        if outcome:
            self.runs[self.run_num].append((self.cracked_num, pwd))

    def output(self, ofile, summary_ofile):
        writer = csv.writer(ofile, delimiter = '\t', quotechar = None,
                            lineterminator = os.linesep)
        average_accum = 0
        for run_idx in range(len(self.runs)):
            hashes = self.runs[run_idx]
            for ahash_num, pwd in hashes:
                average_accum += 1
                writer.writerow([run_idx + 1, ahash_num, pwd if pwd else ''])
        summary_ofile.write(str(average_accum / (len(self.runs) - 1)) +
                            os.linesep)

class Simulator(object):
    def __init__(self, strategy_factory, test_pwds, pwd_priors, tabulator,
                 guess_logger = None):
        self.strategy = strategy_factory(len(test_pwds))
        self.strategy.make_state(pwd_priors)
        self.test_pwds = test_pwds
        self.tabulator = tabulator
        self.prev_pwd = None
        self.guess_number = 0
        self.guess_logger = guess_logger

    def run_step(self):
        action = self.strategy.next_action()
        if not action:
            return True
        self.guess_number += 1
        pwd, idx = action
        if pwd != self.prev_pwd and self.guess_logger is not None:
            self.guess_logger.write('%s\t%s\n' % (pwd, self.guess_number))
        self.prev_pwd = pwd
        outcome = (self.test_pwds[idx] == pwd)
        self.tabulator.record(outcome, pwd)
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
    guess_logger = sys.stdout if args.guess_logger else None
    for _ in range(args.num_runs):
        random.shuffle(test_pwds)
        sim = Simulator(strategy_map[args.strategy], test_pwds,
                        probs, tabulator, guess_logger)
        sim.run(args.num)
        tabulator.reset()
    with open(args.ofile, 'w') as ostream:
        tabulator.output(ostream, args.summary_ofile)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('test_file', help='Input file with test passwords. ')
    parser.add_argument('prob_file', help='Input file with probabilities. ')
    parser.add_argument('strategy', choices = sorted(strategy_map.keys()))
    parser.add_argument('ofile', help='Output file')
    parser.add_argument('--summary-ofile', type=argparse.FileType('w'),
                        default=sys.stdout)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--guess-logger', action='store_true')
    main(parser.parse_args())
