#!/usr/bin/env python

import tensorflow as tf

import unittest

import pass_embedding as pe
import pass_encoder as p_enc

class TestPassEncoder(tf.test.TestCase):
    def test_expand_one_hot(self):
        with self.test_session() as sess:
            config = pe.EmbeddingConfig(alphabet='abc', batch_size=6)
            data_maker = p_enc.encoder_from_config(config)
            data_maker.one_time_tensor_initialize()
            prefix = tf.convert_to_tensor([
              [0, 0],
              [ord('a'), 0],
              [ord('a'), ord('b')],
              [0, 0],
              [ord('b'), 0],
              [ord('b'), ord('c')]])
            labels = tf.convert_to_tensor(
              [ord(c) for c in ['a', 'b', '\n', 'b', 'c', '\n']])
            seq_len = tf.convert_to_tensor([0, 1, 2, 0, 1, 2])
            one_hot_prefix, one_hot_label = data_maker.encode_training(prefix, labels)
            sess.run(data_maker.initializers())

            outputs_out, lab_out, seq_len_out = sess.run(
              [one_hot_prefix, one_hot_label, seq_len])

            self.assertAllClose([[[0, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 0, 0]],
                                 [[1, 0, 0],
                                  [0, 1, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0]],
                                 [[0, 1, 0],
                                  [0, 0, 0]],
                                 [[0, 1, 0],
                                  [0, 0, 1]]],
                                outputs_out)
            self.assertAllClose([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], lab_out)
            self.assertAllEqual([0, 1, 2, 0, 1, 2], seq_len_out)

    def test_input_prefix_to_tensor_padding(self):
        with self.test_session() as sess:
            config = pe.EmbeddingConfig(alphabet='abc')
            maker = p_enc.encoder_from_config(config)
            maker.one_time_tensor_initialize()
            out = maker.encode_inputs([[97, 98, 10],
                                       [99, 99, 10]])
            sess.run(maker.initializers())
            self.assertAllEqual([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 0.]],
                                 [[0., 0., 1.],
                                  [0., 0., 1.],
                                  [0., 0., 0.]]],
                                sess.run(out))



if __name__ == '__main__':
    unittest.main()
