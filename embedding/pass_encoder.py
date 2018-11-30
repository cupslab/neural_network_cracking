import tensorflow as tf


import pass_utils as pm


def output_str_keys(config):
    return [k for k in config.alphabet] + [config.end_of_password_char]


def output_table_from_config(config):
    return NormalTableMaker(config)


def input_table_from_config(config):
    return NormalTableMaker(config)


def output_size(config):
    okeys = output_str_keys(config)
    output_table = output_table_from_config(config)
    return len(set(output_table.make_table(okeys).values()))

def input_size(config):
    ikeys = config.alphabet
    input_table = input_table_from_config(config)
    return len(set(input_table.make_table(ikeys).values()))


class CharacterTableMaker(object):
    def __init__(self, config):
        self.config = config

    def make_table(self, keys):
        raise NotImplementedError()


class NormalTableMaker(CharacterTableMaker):
    def make_table(self, keys):
        value_dict = {}
        for i, key in enumerate(keys):
            value_dict[key] = i

        return value_dict



class BaseInputEncoder(object):
    def __init__(self, config, input_table, output_table):
        self.config = config
        self._initializers = []
        self._input_table = input_table
        self._output_table = output_table

    def one_time_tensor_initialize(self):
        pass

    def initializers(self):
        return self._initializers

    def encode_training(self, inputs, labels):
        return self.encode_inputs(inputs), self.encode_labels_training(labels)

    def encode_testing(self, inputs):
        raise NotImplementedError()

    def encode_to_idx(self, inputs_or_labels):
        raise NotImplementedError()

    def encode_labels(self, labels):
        raise NotImplementedError()

    def encode_inputs(self, inputs):
        raise NotImplementedError()

    def encode_labels_training(self, labels):
        raise NotImplementedError()

    def create_table(self, keys, values, default_value):
        # Use int64 because int32 kernel isn't implemented
        answer = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(
            tf.constant(keys, dtype=tf.int64),
            tf.constant(values, dtype=tf.int64)),
          default_value)
        self._initializers.append(answer.init)
        return answer


class OneHotInputEncoder(BaseInputEncoder):
    def __init__(self, config, input_table, output_table):
        super().__init__(config, input_table, output_table)
        self.input_table = None
        self.output_table = None

    def default_value(self):
        # For one_hot, -1 means a 0 vector
        return -1

    def encode_to_idx(self, inputs_or_labels):
        return self.input_table.lookup(tf.cast(inputs_or_labels, dtype=tf.int64))

    def values_dict(self, keys):
        return self._input_table.make_table(keys)

    def key_values_input(self):
        keys_str = [k for k in self.config.alphabet]

        keys = list(map(ord, keys_str))
        values_dict = self.values_dict(keys_str)
        values = [values_dict[k] for k in keys_str]
        return keys, values

    def key_values_output(self):
        output_keys_str = output_str_keys(self.config)
        output_keys = list(map(ord, output_keys_str))
        output_values_dict = self.values_dict_output(output_keys_str)
        output_values = [output_values_dict[k] for k in output_keys_str]
        return output_keys, output_values

    def values_dict_output(self, keys):
        return self._output_table.make_table(keys)

    def one_time_tensor_initialize(self):
        keys, values = self.key_values_input()
        default_value = self.default_value()

        self.input_table = self.create_table(keys, values, default_value)

        output_keys, output_values = self.key_values_output()
        self.output_table = self.create_table(
          output_keys,
          output_values,
          default_value)

    def encode_inputs(self, inputs):
        input_as_alpha_idx = self.input_table.lookup(
          tf.cast(inputs, dtype=tf.int64))
        return tf.one_hot(
          input_as_alpha_idx,
          input_size(self.config),
          dtype=self._dtype())

    def _dtype(self):
        return pm.data_type_from_string(self.config)

    def encode_labels_training(self, labels):
        return tf.one_hot(
          self.encode_labels(labels),
          output_size(self.config),
          dtype=self._dtype())

    def encode_labels(self, labels):
        return self.output_table.lookup(tf.cast(labels, dtype=tf.int64))

    def encode_testing(self, inputs):
        return self.encode_inputs(inputs)



def encoder_from_config(config):
    input_table = input_table_from_config(config)
    output_table = output_table_from_config(config)
    return OneHotInputEncoder(config, input_table, output_table)
