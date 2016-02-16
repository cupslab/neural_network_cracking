import argparse
import base64
import collections
import csv
import hashlib
import json
import math
import os
import random
import sys

import msgpack

class BloomFilter:
    # BloomFilter code from stackoverflow
    # http://en.wikipedia.org/wiki/Bloom_filter

    def __init__(self, num_bytes, num_probes, iterable=()):
        self.array = bytearray(num_bytes)
        self.num_probes = num_probes
        self.num_bins = num_bytes * 8
        self.num_items = 0
        self.update(iterable)

    def get_probes(self, key):
        # Change to MD5
        # random_fn = random.Random(key).random
        # return (int(random_fn() * self.num_bins) for _ in
        # range(self.num_probes))
        m = hashlib.md5()
        m.update(key.encode('utf8'))
        digest = m.hexdigest()
        output = []
        for i in range(4):
            output.append(int(digest[i * 8:(i+ 1) * 8], 16))
        return [output[-1] % (self.num_bins)]

    def update(self, keys):
        for key in keys:
            self.num_items += 1
            for i in self.get_probes(key):
                self.array[i//8] |= 2 ** (i%8)

    def error(self):
        return (1 - math.exp(-((self.num_probes * self.num_items * 1.0) / self.num_bins)))**self.num_probes

    def __contains__(self, key):
        return all(self.array[i//8] & (2 ** (i%8)) for i in self.get_probes(key))

    def dump_obj(self):
        return bytes(self.array)


def main(args):
    try:
        sys.stderr.write('Treating input as a TSV. \n')
        words = dict([(row[0], int(row[1])) for row in
                      csv.reader(args.ifile, delimiter='\t', quotechar=None)])
    except IndexError:
        sys.stderr.write('Treating input as a line separated file. \n')
        words = dict([(line.strip(os.linesep), rank + 1)
                      for rank, line in enumerate(args.ifile)])
    log_ten_words = collections.defaultdict(list)
    for word in words:
        logten = int(math.log(words[word] + 1, 10))
        for i in range(logten, int(math.log(len(words), 10)) + 1):
            log_ten_words[i].append(word)
    output = {}
    for logten in log_ten_words:
        in_this_bloom = log_ten_words[logten]
        b = BloomFilter(int(len(in_this_bloom) * 1.5) + 1, 1, in_this_bloom)
        print('Size:', len(in_this_bloom), '|', 'Error:', b.error())
        output[str(logten)] = bytes(b.array)
        if args.json:
            output[str(logten)] = base64.b64encode(output[str(logten)]).decode(
                'utf8')
    output = { 'bloom_filter' : output }
    if args.json:
        with open(args.ofile, 'w') as ofile:
            json.dump(output, ofile)
    else:
        with open(args.ofile, 'wb') as ofile:
            ofile.write(msgpack.packb(output))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description=('Create a bloom filter from a list of passwords '
                     'with ranks. '))
    parser.add_argument('-i', '--ifile', type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('-o', '--ofile', required=True)
    parser.add_argument('-j', '--json', action='store_true')
    main(parser.parse_args())
