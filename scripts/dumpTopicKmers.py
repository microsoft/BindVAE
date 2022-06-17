
import numpy as np
import pandas as pd
import sys
import re

def get_topics_strings(topics_words, vocabulary,
                       words_per_topic=100):
  """Returns the summary of the learned topics.

  Arguments:
    topics_words: KxV tensor with topics as rows and words as columns.
    vocabulary: A mapping of word's integer index to the corresponding string.
    words_per_topic: Number of wodrs per topic to return.
  Returns:
    summary: A np.array with strings.
  """
  top_words = np.argsort(-topics_words, axis=1)

  res = []
  for topic_idx in range(topics_words.shape[0]):
      l = ["index={} ".format(topic_idx)]
      for word in top_words[topic_idx, :words_per_topic]:
          #if vocabulary[word].find('N') >= 0:
          #    l += replace_wild_cards(vocabulary[word])
          #else:
          #    l += [vocabulary[word]]
          l += [vocabulary[word]]
      res.append(" ".join(l))

  return np.array(res)


def replace_wild_cards(string):
  res_strs = []
  pp = string.find('N')
  if pp>=0:
      for nuc in ['A','C','G','T']:
          newstr = list(string)
          newstr[pp] = nuc
          res_strs = res_strs + replace_wild_cards("".join(newstr))
  else:
      res_strs = [string]
  return res_strs


vocab_file = sys.argv[1]
beta_file = sys.argv[2]
out_file = sys.argv[3]

vocab = np.load(vocab_file, allow_pickle=True)
words_to_idx = vocab.item()
num_words = len(words_to_idx)
print("#### VOCABULARY SIZE: ",num_words)
vocabulary = [None] * num_words
for word, idx in words_to_idx.items():
    vocabulary[idx] = word

beta = np.load(beta_file)

result = get_topics_strings(beta, vocabulary)
np.savetxt(out_file, result, fmt='%s', delimiter=' ')


