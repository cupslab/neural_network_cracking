import numpy as np
cimport numpy as np

cdef str PASSWORD_END = '\n'

def next_nodes_random_walk(
        self, str astring, double prob,
        np.ndarray[np.double_t, ndim = 1] prediction):
    if len(astring) > 0 and astring[-1] == PASSWORD_END:
        return []
    cdef np.ndarray[np.double_t, ndim = 1] conditional_predictions
    if self.should_make_guesses_rare_char_optimizer:
        conditional_predictions = (
            self.expander.expand_conditional_probs(
                prediction, astring))
    else:
        conditional_predictions = prediction
    cdef np.ndarray[np.double_t, ndim = 1] total_preds = (
        conditional_predictions * prob)
    if len(astring) + 1 > self.max_len:
        if (total_preds[self.pwd_end_idx] >
            self.lower_probability_threshold):
            return [(astring + PASSWORD_END,
                     total_preds[self.pwd_end_idx],
                     conditional_predictions[self.pwd_end_idx])]
    cdef np.ndarray[np.int_t, ndim = 1] indexes = np.arange(len(total_preds))
    cdef np.ndarray above_cutoff = (
        total_preds > self.lower_probability_threshold)
    cdef np.ndarray[np.int_t, ndim = 1] above_indices = indexes[above_cutoff]
    cdef np.ndarray[np.double_t, ndim = 1] probs_above = (
        total_preds[above_cutoff])
    cdef list answer = [0] * len(probs_above)
    cdef int index
    for i in range(len(probs_above)):
        index = above_indices[i]
        answer[i] = (astring + self._chars_list[index], probs_above[i],
                     conditional_predictions[index])
    return answer

def expand_conditional_probs(
        self, np.ndarray[np.double_t, ndim = 1] probs,
        str context, np.ndarray[np.int_t, ndim = 1] expander_cache):
    cdef np.ndarray[np.double_t, ndim = 1] answer = probs[expander_cache]
    cdef bint is_beginning = context == ''
    for i, after_image_char, post_image in self.post_image_idx:
        answer[i] *= self.calc(post_image, after_image_char, is_beginning)
    return answer
