#!/usr/bin/env python

import sys
import argparse
import os
import csv
import collections

class Strategy(object):
    NOT_CRACKED = -1

    def __init__(self, guessing_list, test_set):
        self.guessing_list = guessing_list
        self.test_set = list(test_set)

    def calculate_hash_nums(self):
        raise NotImplementedError

    def output(self, ofile):
        answer = sorted(self.calculate_hash_nums())
        try:
            last_one = (len(answer) - 1) - answer[::-1].index(self.NOT_CRACKED)
            answer = answer[last_one + 1:] + answer[:last_one + 1]
        except ValueError:
            pass
        for value in answer:
            ofile.write(str(value) + '\n')

    @classmethod
    def create(cls, guessing_list, test_set):
        return cls(guessing_list, test_set)

class NaiveStrategy(Strategy):
    def calculate_index_map(self):
        test_map = collections.defaultdict(list)
        for i, pwd in enumerate(self.test_set):
            test_map[pwd].append(i)
        return test_map

    def calculate_hash_nums(self):
        answer = [self.NOT_CRACKED] * len(self.test_set)
        test_map = self.calculate_index_map()
        row_width = len(self.test_set)
        guess_accum = 0
        for i, pwd_tuple in enumerate(self.guessing_list):
            pwd, prob = pwd_tuple
            hit_indices = test_map[pwd] if pwd in test_map else []
            offset, prev_offset = 0, 0
            for index in hit_indices:
                for j in range(prev_offset, index):
                    if answer[j] == self.NOT_CRACKED or j in hit_indices:
                        offset += 1
                prev_offset = index
                answer[index] = guess_accum + offset + 1
            guess_accum += row_width
            row_width -= len(hit_indices)
            assert row_width >= 0
        return answer

class MontyStrat(NaiveStrategy):
    def calculate_hash_nums(self):
        answer = [self.NOT_CRACKED] * len(self.test_set)
        test_map = self.calculate_index_map()
        self.row_width = len(self.test_set)
        self.guess_accum = 0
        accum_prob = 1
        prev_probs = []

        def flush():
            depth = len(prev_probs)
            first_accum = prev_probs[0][2]
            total_hits = []
            previous_not_done = []
            for j, p in enumerate(prev_probs):
                _, prev_pwd, _ = p
                hit_indices = test_map[prev_pwd] if prev_pwd in test_map else []
                previous_not_done.append([0] * len(hit_indices))
                prev, accum = 0, 0
                for l, index in enumerate(hit_indices):
                    for k in range(prev, index):
                        if answer[k] != self.NOT_CRACKED:
                            accum += 1
                    previous_not_done[j][l] = accum
                    prev = index
            for j, p in enumerate(prev_probs):
                _, prev_pwd, _ = p
                hit_indices = test_map[prev_pwd] if prev_pwd in test_map else []
                for l, index in enumerate(hit_indices):
                    answer[index] = first_accum + (
                        depth * (index - previous_not_done[j][l]) + j + 1)
                    total_hits.append((index, j))
                self.row_width -= len(hit_indices)
            total_hits.sort()
            accum = 0
            for item in total_hits:
                index, j = item
                answer[index] -= accum
                accum += len(prev_probs) - j - 1
            self.guess_accum -= accum

        for i, pwd_tuple in enumerate(self.guessing_list):
            pwd, prob = pwd_tuple
            accum_prob -= prob
            real_prob = prob / accum_prob
            assert len(prev_probs) < 10000, 'Running out of memory!'
            if len(prev_probs) > 0 and (real_prob < prev_probs[-1][0]):
                flush()
                prev_probs = []
            prev_probs.append( (real_prob, pwd, self.guess_accum) )
            self.guess_accum += self.row_width
            assert self.row_width >= 0
        if len(prev_probs) > 0:
            flush()
        return answer

strat_map = {
    'naive' : NaiveStrategy,
    'monty' : MontyStrat
}

def read_probability_file(ifname):
    with open(ifname, 'r') as ifile:
        prev_prob = 1
        for row in csv.reader(ifile, delimiter = '\t', quotechar = None):
            prob = float(row[1])
            assert prob <= prev_prob, 'Out of order probability'
            yield (row[0], prob)
            prev_prob = prob

def read_password_file(ifname):
    with open(ifname, 'r') as ifile:
        for line in ifile.readlines():
            yield line.strip(os.linesep)

def main(args):
    strat = strat_map[args.strategy](
        read_probability_file(args.probability_file),
        read_password_file(args.test_password_file))
    strat.output(args.ofile)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Simulate strategies')
    parser.add_argument('strategy', choices = sorted(strat_map.keys()))
    parser.add_argument('test_password_file')
    parser.add_argument('probability_file')
    parser.add_argument('--ofile', default=sys.stdout,
                        type=argparse.FileType('w'))
    main(parser.parse_args())
