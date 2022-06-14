# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import functools
import os

from absl import flags
from absl import logging
import numpy as np
import scipy.sparse
from six.moves import cPickle as pickle
from six.moves import urllib
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


def sparse_matrix_dataset(data_file, num_kmers, shuffle_and_repeat):

  data = []
  for line in open(data_file):
    line = line.rstrip('\n')
    line = line.strip()
    data.append(np.array(line.split()).astype('int'))

  num_documents = len(data)
  print("Finished reading ",num_documents," documents....")

  # Each row is a list of feature ids in the document.
  # A sparse COO matrix is created from this (which automatically sums the repeating features). Then,
  # this COO matrix is converted to CSR format which allows for fast matrix access
  indices = np.array([(row_idx, column_idx)
                      for row_idx, row in enumerate(data)
                      for column_idx in row])
  sparse_matrix = scipy.sparse.coo_matrix(
      (np.ones(indices.shape[0]), (indices[:, 0], indices[:, 1])),
      shape=(num_documents, num_kmers),
      dtype=np.float32)
  sparse_matrix = sparse_matrix.tocsr()

  dataset = tf.data.Dataset.range(num_documents)

  # For training, we shuffle each epoch and repeat the epochs.
  if shuffle_and_repeat:
    dataset = dataset.shuffle(num_documents).repeat()

  # Returns a single document as a dense TensorFlow tensor. The dataset is
  # stored as a sparse matrix outside of the graph.
  def get_row_py_func(idx):
    def get_row_python(idx_py):
      return np.squeeze(np.array(sparse_matrix[idx_py].todense()), axis=0)

    py_func = tf1.py_func(
        get_row_python, [idx], tf.float32, stateful=False)
    py_func.set_shape((num_kmers,))
    return py_func

  dataset = dataset.map(get_row_py_func)
  return dataset


def read_vocabulary(vocab_path):
  vocab = np.load(vocab_path, allow_pickle=True)
  kmers_to_idx = vocab.item()
  num_kmers = len(kmers_to_idx)
  print("[LOG] VOCABULARY SIZE: ",num_kmers)

  vocabulary = [None] * num_kmers
  for word, idx in kmers_to_idx.items():
    vocabulary[idx] = word
  return vocabulary, num_kmers


def build_fake_input_fns(batch_size):
  """Build fake data ."""

  num_words = 100

  random_sample = np.random.randint(
      10, size=(batch_size, num_words)).astype(np.float32)

  def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(random_sample)
    dataset = dataset.batch(batch_size)
    return tf1.data.make_one_shot_iterator(dataset).get_next()

  return test_input_fn


def build_input_fns(train_path, valid_path, test_path, vocab_path, batch_size):
  """Builds iterators for train and evaluation data.
  Returns:
    train_input_fn: A function that returns an iterator over the training data.
    eval_input_fn: A function that returns an iterator over the evaluation data.
    test_input_fn: A function that returns an iterator over the test data.
    vocabulary: A mapping of word's integer index to the corresponding string.
  """

  vocabulary, num_kmers = read_vocabulary(vocab_path)

  # Build an iterator over training data
  def train_input_fn():
    if train_path is None:
        return build_fake_input_fns(batch_size)
    else:
        dataset = sparse_matrix_dataset(train_path, num_kmers, shuffle_and_repeat=True)
        dataset = dataset.batch(batch_size).prefetch(128)
        return tf1.data.make_one_shot_iterator(dataset).get_next()

  # Build an iterator over the heldout set.
  def eval_input_fn():
    if valid_path is None:
        return build_fake_input_fns(batch_size)
    else:
        dataset = sparse_matrix_dataset(valid_path, num_kmers, shuffle_and_repeat=False)
        dataset = dataset.batch(batch_size)
        return tf1.data.make_one_shot_iterator(dataset).get_next()

  def test_input_fn():
    if test_path is None:
        return build_fake_input_fns(batch_size)
    else:
        dataset = sparse_matrix_dataset(test_path, num_kmers, shuffle_and_repeat=False)
        dataset = dataset.batch(batch_size)
        return tf1.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn, test_input_fn, vocabulary

