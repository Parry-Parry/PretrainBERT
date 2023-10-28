from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader 

import torch

def yaml_load(file : str):
    return load(open(file, 'r'), Loader=Loader)

def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=-100):
  """ 
  Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence. 
  * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
  """
  
  device = inputs.device
  labels = inputs.clone()
  
  # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
  probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
  special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
  for sp_id in special_token_indices:
    special_tokens_mask = special_tokens_mask | (inputs==sp_id)
  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  mlm_mask = torch.bernoulli(probability_matrix).bool()
  labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

  # mask  (mlm_probability * (1-replace_prob-orginal_prob))
  mask_prob = 1 - replace_prob - orginal_prob
  mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
  inputs[mask_token_mask] = mask_token_index

  # replace with a random token (mlm_probability * replace_prob)
  if int(replace_prob)!=0:
    rep_prob = replace_prob/(replace_prob + orginal_prob)
    replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[replace_token_mask] = random_words[replace_token_mask]

  # do nothing (mlm_probability * orginal_prob)
  pass

  return inputs, labels, mlm_mask
