import configparser
import json
import logging

import tensorflow as tf

import pass_policy_tensor as ppt
import pass_c_library_loader as pll
import pass_utils as pm
import pass_encoder as p_enc
import pass_policy as pp


class ConfigurationException(Exception):
    pass


_DELIMITER = '\t'

_QUEUE_THRESHOLD = 3


# Based on
# https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py


default_alphabet = (
  'abcdefghijklmnopqrstuvwxyz' +
  'ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
  '0123456789' +
  '!`~@#$%^&*()-_=+[]{}\\|;:\'",<>./?')


class EmbeddingConfig(object):
    def __init__(self,
                 # input params
                 alphabet=default_alphabet,
                 end_of_password_char='\n',
                 enforced_policy='basic',

                 # performance tuning params
                 batch_size=32,
                 init_scale=1.0,
                 num_batch_threads=4,
                 password_batch=5000,
                 password_batch_capacity=50000,
                 logging_freq=1000,
                 data_type='float32',

                 # learning params
                 num_train_epochs=20,
                 learning_rate=0.001,
                 learning_rate_decay='fixed',
                 learning_rate_decay_rate=0.9,
                 learning_rate_decay_period=None,
                 optimizer='grad_descent',

                 # embedding params
                 embedding_window_size=1,
                 embedding_size=16,
                 embedding_num_neg_samples=10,
                 embedding_distortion=0.75):
        """Embedding configuration

    -------------- Example config ---------------
    [embedding]
    num_train_epochs = 1
    learning_rate = 0.001
    embedding_window_size = 5
    embedding_size = 20
    ---------------------------------------------


        alphabet: list of characters definition the alphabet. Default includes
          alpha-numeric characters and keyboard punctuation.

        end_of_password_char: string representing the end of a password. Default is
          the newline character. This character must not be present in alphabet.

        batch_size: integer, number of instances to expect in one batch. Higher
          means storing more data in memory. Default is 32. Higher is faster, but
          has diminishing returns. During training, a higher batch size could make a
          worse model.

        init_scale: float, max values to scale initial values by. Default is 1.0.

        enforced_policy: string, Enforced password policy. See
          pass_policy.py:policy_list for a list of allowed policies. Default is
          'basic'

        password_batch: integer, number of passwords to keep in memory at once.
          Default is 5000.

        data_type: string, what type of floating point computations to do. Allowed
          values: 'float16', 'float32', 'float64'. Default is 'float32'.

        logging_freq: integer, frequency to print logging messages during run loops.
          Higher is less frequent. Default is 1000.

        num_train_epochs: integer, number of epochs to train. Default is 20.

        num_batch_threads: integer, number of threads to read batches. Default is 4.
          More threads make it easier to keep the GPU fully utilized.

        password_batch_capacity: integer, number of prefixes to keep in batch
          capacity. Default is 50000.

        learning_rate: float, training learning rate. Default is 0.001.

        learning_rate_decay: string, allowed values are 'fixed', 'exponential',
          'inverse_time'. Fixed keeps a constant learning rate during training.
          'exponential' uses an exponential decay for the learning rate. Default is
          'fixed'. 'inverse_time' uses decayed_learning_rate = learning_rate / (1 +
          decay_rate * t).

        learning_rate_decay_rate: float, when learning_rate_decay is 'exponential',
          or 'inverse_time' then this value is the decay rate. Default is 0.9.

        learning_rate_decay_period: integer or None. If not provided, then the
          learning rate will decay by learning_rate_decay_rate every epoch. If
          provided, then the learning rate will decay by the rate over that period
          of batches. For example, providing 10 will mean that the learning rate
          will decay by the decay rate every 10 batches.

        optimizer: string, name of the optimizer to use. Options are: 'adadelta',
          'adagrad', 'adam', 'ftrl', 'grad_descent', 'proximal_adagrad', 'rmsprop'.
          Default is 'grad_descent'.

        Embedding implementation is based on
        https://www.tensorflow.org/tutorials/word2vec. The variables in the
        'embedding' section define how to create a character embedding from the
        training data. Configuration values are only valid if use_embedding = True.

        embedding_window_size: integer, the window size to use for creating context
          pairs. Default is 1.

        embedding_size: integer, the number of output indexes for our embedding
          representation. Default is 16.

        embedding_num_neg_samples: integer, number of negative samples to draw
          during training. Default is 10.

        embedding_distortion: float, ratio to distort probabilities by."""

        self.alphabet = str(alphabet)
        if len(self.alphabet) != len(set(alphabet)):
            raise ConfigurationException(
              "Repeated characters in alphabet: " + self.alphabet)

        self.enforced_policy = str(enforced_policy)
        if self.enforced_policy not in pp.policies():
            raise ConfigurationException("Unknown password policy " +
                                         self.enforced_policy)

        self.embedding_window_size = int(embedding_window_size)
        if self.embedding_window_size <= 0:
            raise ConfigurationException("Expected positive embedding_window_size")

        self.embedding_size = int(embedding_size)
        if self.embedding_size <= 0:
            raise ConfigurationException("Expected positive embedding_size")

        self.embedding_num_neg_samples = int(embedding_num_neg_samples)
        if self.embedding_num_neg_samples <= 0:
            raise ConfigurationException(
              "Expected positive embedding_num_neg_samples")

        self.embedding_distortion = float(embedding_distortion)

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ConfigurationException("Expected positive batch_size")

        self.init_scale = float(init_scale)

        self.password_batch = int(password_batch)
        if self.password_batch <= 0:
            raise ConfigurationException("Expected positive password batch")

        self.password_batch_capacity = int(password_batch_capacity)
        if self.password_batch_capacity <= 0:
            raise ConfigurationException("Expected positive password_batch_capacity")

        self.num_batch_threads = int(num_batch_threads)
        if self.num_batch_threads <= 0:
            raise ConfigurationException("Expected positive num_batch_threads")

        self.logging_freq = int(logging_freq)
        if self.logging_freq <= 0:
            raise ConfigurationException("Expected positive logging_freq")

        self.data_type = data_type
        if self.data_type not in pm.data_type_map.keys():
            raise ConfigurationException("Unknown data type " + str(self.data_type))

        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = str(learning_rate_decay)
        if self.learning_rate_decay not in pm.learning_rate_decay_types:
            raise ConfigurationException(
              "Expected %s to be one of %s" % (
                self.learning_rate_decay, pm.learning_rate_decay_types))

        self.learning_rate_decay_rate = float(learning_rate_decay_rate)
        if self.learning_rate_decay_rate <= 0:
            raise ConfigurationException(
              "Expected learning_rate_decay_rate >= 0, not %s" %
              self.learning_rate_decay_rate)

        if learning_rate_decay_period is None:
            self.learning_rate_decay_period = None
        else:
            self.learning_rate_decay_period = int(learning_rate_decay_period)

        self.optimizer = str(optimizer)
        if self.optimizer not in pm.optimizer_map.keys():
            raise ConfigurationException("Unknown optimizer: " + self.optimizer)

        self.num_train_epochs = int(num_train_epochs)
        if self.num_train_epochs <= 0:
            raise ConfigurationException("Expected positive num_train_epochs")

        self.end_of_password_char = str(end_of_password_char)
        if self.end_of_password_char in self.alphabet:
            raise ConfigurationException(
              "End of password char " + self.end_of_password_char +
              " must not be a member of the alphabet: '" + self.alphabet + "'")


    @classmethod
    def from_config_file(cls, cfg_path):
        config = configparser.RawConfigParser()
        config.read([cfg_path])
        vals = {}
        config_defaults = config.defaults()
        if 'embedding' in config:
            config_defaults = config['embedding']

        for key in config_defaults:
            vals[key] = config_defaults[key]

        logging.info("Configuration values %s", vals)
        return cls(**vals)



class EmbeddingTrainer(object):
    def __init__(self, config, model_path=None, tensorboard_logdir=None):
        self._config = config
        self._model_path = model_path

        self._encoder = p_enc.encoder_from_config(config)
        self._loop_runner = pm.LoopRunner(
          self._config.logging_freq,
          tensorboard_logdir=tensorboard_logdir)


    def make_train_graph(self, examples, labels, vocab_counts):
        vocab_size = len(self._config.alphabet)
        emb_dim = self._config.embedding_size
        batch_size = self._config.batch_size

        init_width = self._config.init_scale / emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [vocab_size, emb_dim], -init_width, init_width),
            name="emb")

        sm_w_t = tf.Variable(
          tf.zeros([vocab_size, emb_dim]),
          name="sm_w_t")

        sm_b = tf.Variable(tf.zeros([vocab_size]), name="sm_b")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
          tf.cast(labels, dtype=tf.int64),
          [batch_size, 1])

        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
          true_classes=labels_matrix,
          num_true=1,
          num_sampled=self._config.embedding_num_neg_samples,
          unique=True,
          range_max=vocab_size,
          distortion=self._config.embedding_distortion,
          unigrams=vocab_counts.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b,
                                   [self._config.embedding_num_neg_samples])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits, emb


    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self._config.batch_size
        return nce_loss_tensor


    def optimize(self, loss, samples_per_epoch):
        """Build the graph to optimize the loss function."""

        # Optimizer nodes.
        # Linear learning rate decay.
        global_step = tf.Variable(0, name="global_step")
        optimizer = pm.make_optimizer(
          self._config,
          global_step,
          steps_per_epoch=samples_per_epoch)
        train = optimizer.minimize(
          loss,
          global_step=global_step,
          gate_gradients=optimizer.GATE_NONE)
        return train, global_step

    def preprocess_dataset(self, dataset):
        return (dataset
                .filter(ppt.TensorPasswordFilterer(
                  self._config.alphabet, self._config.enforced_policy))
                .batch(self._config.password_batch))


    def skipgram(self, dataset, randomize=True):
        # Must return (total_samples_processed, examples, labels)
        pwd_batch = (self.preprocess_dataset(dataset)
                     .repeat(self._config.num_train_epochs)
                     .make_one_shot_iterator()
                     .get_next())

        examples, labels = pll.get_library().make_skipgram(
          pwd_batch,
          window_size=self._config.embedding_window_size)

        self._encoder.one_time_tensor_initialize()
        examples_as_idx = self._encoder.encode_to_idx(examples)
        labels_as_idx = self._encoder.encode_to_idx(labels)

        ex, lbl = tf.train.batch(
          [examples_as_idx, labels_as_idx],
          batch_size=self._config.batch_size,
          capacity=self._config.password_batch_capacity,
          enqueue_many=True,
          allow_smaller_final_batch=False)

        if not randomize:
            return ex, lbl

        return tf.train.shuffle_batch(
          [ex, lbl],
          batch_size=self._config.batch_size,
          num_threads=self._config.num_batch_threads,
          capacity=self._config.password_batch_capacity,
          enqueue_many=True,
          allow_smaller_final_batch=False,
          min_after_dequeue=int(
            self._config.password_batch_capacity / _QUEUE_THRESHOLD))



    def initial_counts(self, dataset, session):
        # Must return (counts, samples_per_epoch)
        counts = tf.get_variable(
          "histogram_char_counts",
          [len(self._config.alphabet)],
          dtype=tf.int64,
          initializer=tf.zeros_initializer)
        samples = tf.get_variable(
          "samples_per_epoch",
          (),
          dtype=tf.int64,
          initializer=tf.zeros_initializer)

        pwd_batch = (self.preprocess_dataset(dataset)
                     .make_one_shot_iterator()
                     .get_next())

        char_counts, samples_count = pll.get_library().character_counts(
          pwd_batch,
          alphabet=self._config.alphabet,
          window_size=self._config.embedding_window_size)

        update_counts = tf.assign_add(counts, char_counts)
        update_samples = tf.assign_add(samples, samples_count)

        session.run([counts.initializer, samples.initializer])

        def _run_fn(_):
            session.run([update_counts, update_samples])

        logging.info('Starting to count batches per epoch')
        self._loop_runner.run_loop(_run_fn)

        count_hist, samples_per_epoch = session.run([counts, samples])
        logging.info('Number of samples per epoch %s', samples_per_epoch)
        return count_hist, samples_per_epoch


    def build_graph(self, dataset, session):
        logging.info('Reading embedding training data for preprocessing...')
        vocab_counts, samples_per_epoch = self.initial_counts(dataset, session)

        logging.info('Building embedding optimizer...')
        examples, labels = self.skipgram(dataset)

        true_logits, sampled_logits, emb = self.make_train_graph(
          examples, labels, vocab_counts)
        loss = self.nce_loss(true_logits, sampled_logits)
        optimize, global_step = self.optimize(loss, samples_per_epoch)
        return optimize, loss, global_step, emb


    def _save(self, session, saver):
        if self._model_path:
            logging.info('Saving embedding model to %s', self._model_path)
            saver.Save(session, self._model_path)


    def train(self, dataset, session):
        optimize, loss, global_step, emb = self.build_graph(
          dataset, session)

        reset_mean_loss, update_mean_loss, train_loss_summary = pm.running_mean(
          loss,
          'embedding_loss',
          batch_size=self._config.batch_size)

        logging.info('Starting embedding training...')
        session.run([tf.global_variables_initializer(),
                     tf.local_variables_initializer()])
        session.run(self._encoder.initializers())

        saver = tf.train.Saver()

        def _run_fn(_):
            session.run([optimize, update_mean_loss])

        def _logging_fn(runner):
            loss_sum, glob_step_out = session.run([train_loss_summary, global_step])
            runner.write_summary(loss_sum, glob_step_out)
            session.run(reset_mean_loss)

        self._loop_runner.run_loop(_run_fn, _logging_fn)

        self._save(session, saver)
        emb = session.run(emb)
        return emb


    def train_and_save(self, dataset, session, output_file):
        emb = self.train(dataset, session)
        out_data = {}
        for i in range(emb.shape[0]):
            out_data[self._config.alphabet[i]] = emb[i].tolist()

        json.dump(out_data, output_file)


class CharEmbeddingLoader(object):
    class LoadException(Exception):
        def __init__(self, msg):
            super().__init__('Error while loading embedding: %s' % msg)

    def __init__(self, config):
        self._config = config

    def read_from_file(self, input_file):
        input_field = json.load(input_file)

        output = {}
        for key_char, value in input_field.items():
            if key_char in output:
                raise self.LoadException('Key duplicated in load file %s' % key_char)

            output[key_char] = value

        output_keys_uniq = set(output.keys())
        alphabet_uniq = set(self._config.alphabet)
        if set(output.keys()) != set(self._config.alphabet):
            raise self.LoadException(
              ('Embedding file alphabet does not equal expected alphabet. '
               'Found %s. Expected %s') % (alphabet_uniq, output_keys_uniq))

        output_as_float = {}
        for key, value in output.items():
            if len(value) != self._config.embedding_size:
                raise self.LoadException(
                  'Expected %d elements in embedding file. Found %d' % (
                    self._config.embedding_size,
                    len(value)))

            converted = []
            for v in value:
                try:
                    converted.append(float(v))
                except ValueError as e:
                    raise self.LoadException('Failed to convert %s to float. %s' % (v, e))

            output_as_float[key] = converted

        return output_as_float


class EmbeddingAnalogy(object):
    def __init__(self, config, embedding):
        self._config = config

        self._alpha_to_idx = {}
        self.embedding = []
        for i, char in enumerate(self._config.alphabet):
            self._alpha_to_idx[char] = i
            self.embedding.append(embedding[char])

        self._analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        self._analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        self._analogy_c = tf.placeholder(dtype=tf.int32) # [N]


    def _build_eval_graph(self, analogy):
        analogy_a, analogy_b, analogy_c = analogy
        nemb = tf.nn.l2_normalize(self.embedding, 1)
        a_emb = tf.gather(nemb, analogy_a)
        b_emb = tf.gather(nemb, analogy_b)
        c_emb = tf.gather(nemb, analogy_c)

        target = c_emb + (b_emb - a_emb)

        dist = tf.matmul(target, nemb, transpose_b=True)
        _, pred_idx = tf.nn.top_k(dist, 4)
        return pred_idx

    def _build_analogy_input(self):
        return self._build_eval_graph(
          (self._analogy_a, self._analogy_b, self._analogy_c))

    def run_analogies(self, analogies, session):
        analogy_str_a, analogy_str_b, analogy_str_c = analogies

        analogy_idx_a = [self._alpha_to_idx[c] for c in analogy_str_a]
        analogy_idx_b = [self._alpha_to_idx[c] for c in analogy_str_b]
        analogy_idx_c = [self._alpha_to_idx[c] for c in analogy_str_c]
        answer = self._build_analogy_input()
        return session.run(answer, feed_dict={
          self._analogy_a : analogy_idx_a,
          self._analogy_b : analogy_idx_b,
          self._analogy_c : analogy_idx_c
        })
