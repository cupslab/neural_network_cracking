# pwd_guess_ctypes.pyx

import numpy as np
cimport numpy as np

PASSWORD_END = '\n'

def next_nodes(self, str astring, double prob, np.ndarray prediction):
    total_preds = prediction * prob
    cdef double chain_prob
    cdef str chain_pass
    for i, char in enumerate(self.chars_list):
        chain_prob = total_preds[i]
        if chain_prob < self.lower_probability_threshold:
            continue
        chain_pass = astring + char
        if char == PASSWORD_END:
            self.output_serializer.serialize(chain_pass, chain_prob)
            self.generated += 1
        elif len(chain_pass) > self.max_len:
            continue
        elif char != PASSWORD_END:
            yield chain_pass, chain_prob
