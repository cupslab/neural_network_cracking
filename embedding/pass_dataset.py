import logging
import os.path

import tensorflow as tf


class PasswordDatasetMaker(object):
    def __init__(self, input_paths):
        self.input_paths = input_paths


class PasswordDatasetMakerInMemory(PasswordDatasetMaker):
    def make(self):
        for ipath in self.input_paths:
            if not os.path.exists(ipath):
                raise ValueError('Path %s does not exist' % ipath)

        data = []
        logging.info('Reading all input data into memory')
        for ipath in self.input_paths:
            logging.info('Reading from %s', ipath)
            with open(ipath, 'r') as ifile:
                for line in ifile:
                    data.append(line.rstrip('\n'))

            logging.info('Done reading from %s', ipath)

        logging.info('Creating tensor from dataset')
        with tf.device('/cpu:0'):
            data_as_tensor = tf.convert_to_tensor(data)


        logging.info('Done reading input into memory')
        return tf.data.Dataset.from_tensor_slices(data_as_tensor)


class PasswordDatasetMakerFromFile(PasswordDatasetMaker):
    def make(self):
        return tf.data.TextLineDataset(self.input_paths)
