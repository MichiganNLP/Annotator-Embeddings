# src: https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html
import torch
import numpy as np
from transformers import LogitsProcessor


def set_scores_to_inf_for_banned_tokens(scores, banned_tokens):
    """
    Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
    list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

    Args:
        scores: logits distribution of shape (batch size, vocabulary size)
        banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return scores

    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))

    banned_mask = (
        torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    )
    scores = scores.masked_fill(banned_mask, -float("inf"))
    return scores


class ConstraintLogits(LogitsProcessor):
  def __init__(self, tokenizer, constrained_tokens):
    """
    vocab is a dictionary where the keys are tokens
    and the values are the corresponding ids.
    """
    # create an array of tokens
    # remove the 'Ġ' token (used to represent a blank space in the tokenizer)
    self.keys = list(tokenizer.vocab.keys())
    if 'Ġ' in self.keys:
        index_to_pop = self.keys.index('Ġ') 
        self.keys.pop(index_to_pop)
    self.keys = np.array(self.keys)

    # create an array of ids
    # also remove the 'Ġ' token
    self.values = list(tokenizer.vocab.values())
    if 'Ġ' in self.keys:
        self.values.pop(index_to_pop)
    self.values = np.array(self.values)

    # vectorized function used to get the token
    # ignores leading whitespaces and 'Ġ' tokens
    current_token = lambda x: x.strip('Ġ ')
    self.current_token = np.vectorize(current_token)

    # get the indexes of all IDs that are not in the constrained tokens
    not_constrained_indexes = np.where(np.invert(np.isin(self.current_token(self.keys), list(constrained_tokens))))

    # create sets of tokens that are not in the constrained tokens
    self.not_constrained_indexes = self.values[not_constrained_indexes]

  def __call__(self, input_ids, scores):
    banned_tokens = []
    # for every beam (partially generated sentence)
    for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
        banned_tokens.append(self.not_constrained_indexes)
    # set the scores of all banned tokens over the beams to -inf
    scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
    return scores
