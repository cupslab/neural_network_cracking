#!/usr/bin/env python

import unittest
import tempfile

import tensorflow as tf
import io


import pass_embedding as pe

class TestEmbeddingTrainer(tf.test.TestCase):
    def test_initial_counts_full_batch(self):
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'pass',
          'word',
          'db'
        ])

        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          password_batch=5,
          embedding_window_size=1)
        emb_trainer = pe.EmbeddingTrainer(config)
        with self.test_session() as session:
            counts, samples = emb_trainer.initial_counts(test_dataset, session)

        #         a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t  u  v
        expect = [1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0,
        #         w  x  y  z
                  1, 0, 0, 0]

        self.assertAllEqual(expect, counts)
        self.assertEqual(7, samples)

    def test_initial_counts_partial_batch(self):
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'pass',
          'word',
          'db'
        ])

        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          password_batch=1,
          embedding_window_size=1)
        emb_trainer = pe.EmbeddingTrainer(config)
        with self.test_session() as session:
            counts, samples = emb_trainer.initial_counts(test_dataset, session)

        #         a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t  u  v
        expect = [1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0,
        #         w  x  y  z
                  1, 0, 0, 0]

        self.assertAllEqual(expect, counts)
        self.assertEqual(7, samples)

    def test_initial_counts_partial_larger_window(self):
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'pass',
          'word',
          'db'
        ])

        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          password_batch=1,
          embedding_window_size=2)
        emb_trainer = pe.EmbeddingTrainer(config)
        with self.test_session() as session:
            counts, samples = emb_trainer.initial_counts(test_dataset, session)

        #         a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t  u  v
        expect = [1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0,
        #         w  x  y  z
                  1, 0, 0, 0]

        self.assertAllEqual(expect, counts)
        self.assertEqual(8, samples)


    def test_initial_counts_partial_largerer_window(self):
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'passj',
          'word',
          'db'
        ])

        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          password_batch=1,
          embedding_window_size=3)
        emb_trainer = pe.EmbeddingTrainer(config)
        with self.test_session() as session:
            counts, samples = emb_trainer.initial_counts(test_dataset, session)

        #         a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t  u  v
        expect = [1, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0,
        #         w  x  y  z
                  1, 0, 0, 0]

        self.assertAllEqual(expect, counts)
        self.assertEqual(9, samples)

    def test_skipgram(self):
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'passj',
          'word',
          'db'
        ])

        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          password_batch=5,
          batch_size=10,
          embedding_window_size=3)
        emb_trainer = pe.EmbeddingTrainer(config)
        examples, labels = emb_trainer.skipgram(test_dataset, randomize=False)

        with self.test_session() as session:
            session.run([tf.tables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                ex_out, lbl_out = session.run([examples, labels])

            except tf.errors.OutOfRangeError:
                pass

            finally:
                coord.request_stop()

            coord.join(threads)

        self.assertAllEqual([18, 18, 18, 9, 9, 9, 3, 3, 3, 18], ex_out)
        self.assertAllEqual([15, 0, 18, 0, 18, 18, 22, 14, 17, 15], lbl_out)


    def test_skipgram_randomize(self):
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'passj',
          'word',
          'db'
        ])

        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          password_batch=5,
          batch_size=10,
          embedding_window_size=3)
        emb_trainer = pe.EmbeddingTrainer(config)
        examples, labels = emb_trainer.skipgram(test_dataset, randomize=True)

        with self.test_session() as session:
            session.run([tf.tables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                session.run([examples, labels])

            except tf.errors.OutOfRangeError:
                pass

            finally:
                coord.request_stop()

            coord.join(threads)

    def test_graph_builds(self):
        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          password_batch=5,
          embedding_window_size=3)
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'passj',
          'word',
          'db'
        ])
        emb_trainer = pe.EmbeddingTrainer(config)
        with self.test_session() as session:
            emb_trainer.build_graph(test_dataset, session)

    def test_train_loop(self):
        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          password_batch=5,
          embedding_window_size=1,
          batch_size=2,
          embedding_size=4,
          embedding_num_neg_samples=2,
          logging_freq=1,
          num_train_epochs=1)
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'passj',
          'word',
          'db'
        ])
        emb_trainer = pe.EmbeddingTrainer(config)
        with self.test_session() as session:
            answer = emb_trainer.train(test_dataset, session)

        self.assertAllEqual([26, 4], answer.shape)

    def test_train_save(self):
        with tempfile.NamedTemporaryFile(mode='w') as ofile:
            config = pe.EmbeddingConfig(
              alphabet='abcdefghijklmnopqrstuvwxyz',
              password_batch=5,
              embedding_window_size=1,
              batch_size=2,
              embedding_size=4,
              embedding_num_neg_samples=2,
              logging_freq=1,
              num_train_epochs=1)
            test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
              'passj',
              'word',
              'db'
            ])

            emb_trainer = pe.EmbeddingTrainer(config)
            with self.test_session() as session:
                emb_trainer.train_and_save(test_dataset, session, ofile)

            ofile.flush()


class TestAnalogyEval(tf.test.TestCase):
    def test_train_and_load_embedding(self):
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'passj',
          'word',
          'db'
        ])

        with tempfile.NamedTemporaryFile(mode='w') as ofile:
            config = pe.EmbeddingConfig(
              alphabet='abcdefghijklmnopqrstuvwxyz',
              password_batch=5,
              embedding_window_size=1,
              batch_size=2,
              embedding_size=4,
              embedding_num_neg_samples=2,
              logging_freq=1,
              num_train_epochs=1)
            emb_trainer = pe.EmbeddingTrainer(config)
            with self.test_session() as session:
                emb_trainer.train_and_save(test_dataset, session, ofile)

            ofile.flush()

            with open(ofile.name, 'r') as ifile:
                loader = pe.CharEmbeddingLoader(config)
                output = loader.read_from_file(ifile)

        with self.test_session() as session:
            evaler = pe.EmbeddingAnalogy(config, output)
            evaler.run_analogies(('a', 'b', 'c'), session)


class TestEmbeddingLoader(tf.test.TestCase):
    def test_train_and_load_embedding(self):
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices([
          'passj',
          'word',
          'db'
        ])

        with tempfile.NamedTemporaryFile(mode='w') as ofile:
            config = pe.EmbeddingConfig(
              alphabet='abcdefghijklmnopqrstuvwxyz',
              password_batch=5,
              embedding_window_size=1,
              batch_size=2,
              embedding_size=4,
              embedding_num_neg_samples=2,
              logging_freq=1,
              num_train_epochs=1)
            emb_trainer = pe.EmbeddingTrainer(config)
            with self.test_session() as session:
                emb_trainer.train_and_save(test_dataset, session, ofile)

            ofile.flush()

            with open(ofile.name, 'r') as ifile:
                loader = pe.CharEmbeddingLoader(config)
                output = loader.read_from_file(ifile)

                self.assertEqual(26, len(output))
                for _, value in output.items():
                    self.assertEqual(4, len(value))

    def test_alphabet_not_equal(self):
        config = pe.EmbeddingConfig(
          alphabet='abcdefghijklmnopqrstuvwxyz',
          embedding_size=1)
        ifile = io.StringIO("""{ "a" : [0.4], "b" : [0.1]}""")
        loader = pe.CharEmbeddingLoader(config)
        try:
            loader.read_from_file(ifile)
            errored = False
        except pe.CharEmbeddingLoader.LoadException:
            errored = True

        self.assertTrue(errored)

    def test_alphabet_not_equal_extra(self):
        ifile = io.StringIO("""{ "a" : [0.4], "b" : [0.1], "c" : [0.2]}""")
        config = pe.EmbeddingConfig(alphabet='ab', embedding_size=1)
        loader = pe.CharEmbeddingLoader(config)
        try:
            loader.read_from_file(ifile)
            errored = False
        except pe.CharEmbeddingLoader.LoadException:
            errored = True

        self.assertTrue(errored)

    def test_alphabet_size_not_equal(self):
        ifile = io.StringIO("""{ "a": [0.4], "b" : [0.1, 0.4], "c": [0.2] }""")
        config = pe.EmbeddingConfig(alphabet='ab', embedding_size=1)
        loader = pe.CharEmbeddingLoader(config)
        try:
            loader.read_from_file(ifile)
            errored = False
        except pe.CharEmbeddingLoader.LoadException:
            errored = True

        self.assertTrue(errored)

    def test_alphabet_not_float(self):
        ifile = io.StringIO("""{ "a": [0.4], "b" : ["asdf"], "c" : [0.2] }""")
        config = pe.EmbeddingConfig(alphabet='ab', embedding_size=1)
        loader = pe.CharEmbeddingLoader(config)
        try:
            loader.read_from_file(ifile)
            errored = False
        except pe.CharEmbeddingLoader.LoadException:
            errored = True

        self.assertTrue(errored)


if __name__ == '__main__':
    unittest.main()
