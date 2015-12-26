import numpy as np

PASSWORD_END = '\n'

def next_nodes_random_walk_tuple(self, astring, prob, prediction):
    if len(astring) > 0 and astring[-1] == PASSWORD_END:
        return []
    astring_len = sum(map(len, astring))
    if self.should_make_guesses_rare_char_optimizer:
        conditional_predictions = (
            self.expander.expand_conditional_probs(
                prediction, astring))
    else:
        conditional_predictions = prediction
    total_preds = conditional_predictions * prob
    if astring_len + 1 > self.max_len:
        if (total_preds[self.pwd_end_idx] >
            self.lower_probability_threshold):
            return [(astring + (PASSWORD_END,),
                     total_preds[self.pwd_end_idx],
                     conditional_predictions[self.pwd_end_idx])]
    indexes = np.arange(len(total_preds))
    above_cutoff = total_preds > self.lower_probability_threshold
    above_indices = indexes[above_cutoff]
    probs_above = total_preds[above_cutoff]
    answer = [0] * len(probs_above)
    for i in range(len(probs_above)):
        index = above_indices[i]
        answer[i] = (astring + (self._chars_list[index],), probs_above[i],
                     conditional_predictions[index])
    return answer

def expand_conditional_probs(self, probs, context, expander_cache):
    answer = probs[expander_cache]
    for i, after_image_char, post_image in self.post_image_idx:
        answer[i] *= self.calc(post_image, after_image_char, context)
    return answer
