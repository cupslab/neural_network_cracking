import tensorflow as tf

import time
import logging



data_type_map = {
  'float16' : tf.float16,
  'float32' : tf.float32,
  'float64' : tf.float64
}


def data_type_from_string(config):
    return data_type_map[config.data_type]


optimizer_map = {
  'adadelta' : tf.train.AdadeltaOptimizer,
  'adagrad' : tf.train.AdagradOptimizer,
  'adam' : tf.train.AdamOptimizer,
  'ftrl' : tf.train.FtrlOptimizer,
  'grad_descent' : tf.train.GradientDescentOptimizer,
  'proximal_adagrad' : tf.train.ProximalAdagradOptimizer,
  'rmsprop' : tf.train.RMSPropOptimizer
}

learning_rate_decay_types = ['fixed', 'exponential', 'inverse_time']


def make_optimizer(config, global_step, steps_per_epoch=None):
    config_lr_decay_period = config.learning_rate_decay_period
    if config_lr_decay_period is None:
        assert steps_per_epoch is not None or config.learning_rate_decay == 'fixed'
        learning_rate_decay_period = steps_per_epoch
    else:
        learning_rate_decay_period = config_lr_decay_period


    if config.learning_rate_decay == 'fixed':
        learning_rate = tf.convert_to_tensor(config.learning_rate)

    elif config.learning_rate_decay == 'exponential':
        learning_rate = tf.train.exponential_decay(
          config.learning_rate,
          global_step,
          learning_rate_decay_period,
          config.learning_rate_decay_rate)

    elif config.learning_rate_decay == 'inverse_time':
        learning_rate = tf.train.inverse_time_decay(
          config.learning_rate,
          global_step,
          learning_rate_decay_period,
          config.learning_rate_decay_rate)

    else:
        raise ValueError('Unknown learning_rate_decay %s' %
                         config.learning_rate_decay)

    tf.summary.scalar('learning_rate', learning_rate)
    return optimizer_map[config.optimizer](learning_rate)




def running_mean(cost, tag_name, batch_size=1):
    with tf.name_scope("running_mean_" + tag_name):
        with tf.variable_scope(tag_name):
            cost_sum = tf.get_variable(
              "cost_sum",
              initializer=tf.zeros_initializer,
              dtype=tf.float64,
              shape=(),
              collections=[tf.GraphKeys.LOCAL_VARIABLES],
              trainable=False)
            batches = tf.get_variable(
              "cost_num_batches",
              initializer=tf.zeros_initializer,
              dtype=tf.int32,
              shape=(),
              collections=[tf.GraphKeys.LOCAL_VARIABLES],
              trainable=False)

        cost_add = tf.assign_add(cost_sum, tf.cast(cost, dtype=tf.float64))
        batches_add = tf.assign_add(batches, batch_size)
        update_cost_mean = tf.group(cost_add, batches_add)

        reset_batches = tf.assign(batches, 0)
        reset_cost_sum = tf.assign(cost_sum, 0.0)
        reset_cost_mean = tf.group(reset_batches, reset_cost_sum)

        mean_cost = tf.divide(
          cost_sum,
          tf.cast(batches, dtype=tf.float64))
        train_loss_summary = tf.summary.scalar(tag_name, mean_cost)

    return reset_cost_mean, update_cost_mean, train_loss_summary



class LoopRunner(object):
    def __init__(self, logging_freq=1000, tensorboard_logdir=None):
        if tensorboard_logdir is not None:
            self._writer = tf.summary.FileWriter(tensorboard_logdir)
        else:
            self._writer = None

        self._logging_freq = logging_freq

    def write_summary(self, oper, global_step):
        if self._writer is not None:
            self._writer.add_summary(oper, global_step=global_step)

    def run_loop(self, step_fn, logging_fn=None):
        start_time = time.time()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 0
        logging_counter = 0
        logging_freq = self._logging_freq

        try:
            while not coord.should_stop():
                step_fn(self)
                step += 1
                logging_counter += 1
                if logging_counter == logging_freq:
                    if logging_fn is not None:
                        logging_fn(self)

                    logging.info(
                      'Step %d, %.3f secs/batch',
                      step,
                      (time.time() - start_time) / step)
                    logging_counter = 0

        except tf.errors.OutOfRangeError:
            pass

        finally:
            coord.request_stop()
            coord.join(threads)

        logging.info('Time taken %.0f seconds', time.time() - start_time)
        return step
