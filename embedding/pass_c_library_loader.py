import sys
import os
import logging
import tensorflow as tf

try_dir = os.path.join(
  os.path.dirname(os.path.realpath(__file__)),
  'expand_operation.so')
try:
    custom_library = tf.load_op_library(try_dir)
    logging.info('Loaded external library: %s', try_dir)

# pylint: disable=broad-except
# because we don't
# really know what exception might by thrown and it doesn't matter
except Exception as e:
    logging.fatal('Not able to load binary library for expand_operation. %s - %s',
                  try_dir, str(e))
    sys.exit(1)


assert custom_library is not None

def get_library():
    return custom_library
