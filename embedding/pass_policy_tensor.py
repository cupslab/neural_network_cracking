import pass_policy as ppol
import pass_c_library_loader as pll

import tensorflow as tf
import numpy as np


_DEFAULT_ENCODING = 'utf8'


class TensorPasswordPolicy(object):
    def __init__(self, policy_name):
        self._policy = ppol.get_policy(policy_name)

    def __call__(self, tensor_pwd):
        def _fn_one_dimension(pwd_arr):
            output = np.zeros_like(pwd_arr, dtype=np.bool_)
            for i, pwd in enumerate(pwd_arr):
                output[i] = self._policy.pwd_complies(pwd.decode(_DEFAULT_ENCODING))

            return output

        tensor_pwd = tf.convert_to_tensor(tensor_pwd)
        answer = tf.py_func(
          _fn_one_dimension,
          [tensor_pwd],
          tf.bool,
          stateful=False)
        answer.set_shape(tensor_pwd.shape)
        return answer


class TensorPasswordFilterer(object):
    def __init__(self, alphabet, enforced_policy, uniquify=False):
        self._filterer = ppol.PasswordFilterer(alphabet, enforced_policy)
        self._seen = set()
        self._uniquify = uniquify
        self._number_removed = 0
        self._policy = enforced_policy
        self._alphabet = alphabet

    def _fn_one_pwd_uniquify(self, pwd):
        if pwd in self._seen:
            self._number_removed += 1
            return False

        self._seen.add(pwd)
        return self._filterer(pwd.decode(_DEFAULT_ENCODING))

    def _fn_one_dim_uniquify(self, pwd_arr):
        output = np.zeros_like(pwd_arr, dtype=np.bool_)
        for i, pwd in enumerate(pwd_arr):
            output[i] = self._fn_one_pwd_uniquify(pwd)

        return output

    def __call__(self, tensor_pwd):
        tensor_pwd = tf.convert_to_tensor(tensor_pwd)
        if self._uniquify:
            function_to_call = self._fn_one_dim_uniquify
            if len(tensor_pwd.shape.dims) == 0:
                function_to_call = self._fn_one_pwd_uniquify

            answer = tf.py_func(
              function_to_call,
              [tensor_pwd],
              tf.bool,
              stateful=self._uniquify)
            answer.set_shape(tensor_pwd.shape)
            return answer

        else:
            return pll.get_library().pass_policy_filter(
              tensor_pwd,
              alphabet=self._alphabet,
              policy=self._policy)

    def reset(self):
        self._number_removed = 0
        self._filterer.reset()

    @property
    def number_passed(self):
        return self._filterer.number_passed

    @property
    def number_removed(self):
        return self._filterer.number_removed + self._number_removed


def filterer_from_config(config, uniquify=False):
    return TensorPasswordFilterer(
      config.alphabet, config.enforced_policy, uniquify=uniquify)
