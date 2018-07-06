import keras
import numpy as np
from itertools import islice
import string

class DataGenerator(keras.utils.Sequence):
    def __init__(self, fname, batch_size, max_len):
        self.fname = fname
        self.batch_size = batch_size
        self.len = None
        self.current_pwd_position = 0
        self.max_len = max_len
        self.buffer_line_segments = None
        self.char_bag = ('\n' + string.ascii_lowercase + string.ascii_uppercase +
                    string.digits + '~!@#$%^&*(),.<>/?\'"{}[]\\|-_=+;: `')
        with open(fname) as f:
            self.pwd_lines = [x for x in f.readlines() if self.pwd_filter(x)]
        print("File {} num passwords = {}".format(fname, len(self.pwd_lines)))

        self.vocab = self.create_vocab()

    def __len__(self):
        if self.len is None:
            self.len = int(np.floor(self.get_file_lines(self.fname)/self.batch_size))
            return self.len
        else:
            return self.len

    def __getitem__(self, idx):
        lines = []
        if self.buffer_line_segments:
            for line in self.buffer_line_segments:
                lines.append(line)
            self.buffer_line_segments = []
        i = len(lines)
        # while 1:
        #    line = self.file.readline()
        lines_read = 0
        for line in islice(self.pwd_lines, self.current_pwd_position, None):
            lines_read += 1
            if len(line) > self.max_len:
                chunk_lines = [line[i:i + self.max_len] for i in range(0, len(line), self.max_len)]
                #gen = self.chunks(line, self.max_len)
                for j in range(len(chunk_lines)):
                    lines.append(chunk_lines[j])
                    i += 1
                    if i == self.batch_size:
                        self.buffer_line_segments = chunk_lines[j+1:]
                        self.current_pwd_position += lines_read
                        return self.convert_to_arr(lines)
            else:
                lines.append(line)
                i += 1
                if i == self.batch_size:
                    self.current_pwd_position += lines_read
                    return self.convert_to_arr(lines)

    def on_epoch_end(self):
        #self.file.close()
        #self.file = open(self.fname, "r")
        pass

    def convert_to_arr(self, lines):
        lines = [line + ('\n' * (self.max_len - len(line))) for line in lines]
        arr_X = np.zeros(shape=(self.batch_size, self.max_len), dtype=np.int8)
        arr_Y = np.zeros(shape=(self.batch_size, self.max_len, len(self.vocab)), dtype=np.int8)
        for i, line in enumerate(lines):
            shifted_line = line[1:] + '\n'
            for j, char in enumerate(line):
                arr_X[i, j] = self.vocab[char]
            for j, char in enumerate(shifted_line):
                arr_Y[i, j, self.vocab[char]] = 1
        return (arr_X, arr_Y)

    def get_file_lines(self, file):
        with open(file) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def pwd_filter(self, pwd):
        return all([c in self.char_bag for c in pwd])

    def create_vocab(self):
        return {c: i for i, c in enumerate(self.char_bag)}

    def get_vocab(self):
        return self.vocab