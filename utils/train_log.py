import sys
import argparse
import csv
import datetime

class LogFile(object):
    def __init__(self, ifname, ostream):
        self.generations = {}
        self.ifname = ifname
        self.start_time = None
        self.writer = csv.writer(ostream)

    def process_line(self, line):
        tokens = line.split(' ')
        timestamp_string = ' '.join(tokens[:2]) + '0' * 3
        # Looks like: 2015-11-05 08:59:30,439
        timestamp = datetime.datetime.strptime(timestamp_string,
                                               '%Y-%m-%d %H:%M:%S,%f')
        if self.start_time is None:
            self.start_time = timestamp
        offset = (timestamp - self.start_time).total_seconds()
        if tokens[3] == 'Train' and tokens[4] == 'loss':
            tr_loss, test_loss = float(tokens[5][:-1]), float(tokens[8][:-1])
            accuracy = float(tokens[11][:-1])
            self.writer.writerow([
                self.generation, self.chunk,
                tr_loss, test_loss, accuracy, offset])
        if tokens[3] == 'Chunk':
            self.chunk = int(tokens[4][:-1])
        if tokens[3] == 'Generation' and tokens[4] != 'accuracy:':
            self.generation = int(tokens[4])
        return

    def read(self):
        with open(self.ifname, 'r') as lgf:
            for line in lgf:
                self.process_line(line.strip('\n'))

def main(args):
    lg = LogFile(args.ifile, args.ofile)
    lg.read()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('ifile', help = 'Input file. ')
    parser.add_argument('-o', '--ofile', type = argparse.FileType('w'),
                        help = 'Input file. Default is stdout. ',
                        default = sys.stdout)
    main(parser.parse_args())
