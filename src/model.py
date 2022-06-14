# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import functools
import os
import pandas as pd

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

def _clip_dirichlet_parameters(x):
  """Clips Dirichlet param for numerically stable KL and nonzero samples."""
  return tf.clip_by_value(x, .1, 1e3)


def make_encoder(activation, num_topics, layer_sizes):
  """Create the encoder function.
  Args:
    activation: Activation function to use.
    num_topics: The number of topics.
    layer_sizes: The number of hidden units per layer in the encoder.

  Returns:
    encoder: A `callable` mapping a bag-of-kmers `Tensor` to a
      `tfd.Distribution` instance over topics.
  """
  encoder_net = tf.keras.Sequential()
  for num_hidden_units in layer_sizes:
    encoder_net.add(
        tf.keras.layers.Dense(
            num_hidden_units,
            activation=activation,
            kernel_initializer=tf.initializers.GlorotNormal()))
  encoder_net.add(
      tf.keras.layers.Dense(
          num_topics,
          activation=lambda x: _clip_dirichlet_parameters(tf.nn.softplus(x)),
          kernel_initializer=tf.initializers.GlorotNormal()))

  def encoder(bag_of_kmers):
    with tf.name_scope("encoder"):
      return tfd.Dirichlet(concentration=encoder_net(bag_of_kmers),
                           name="topics_posterior")

  return encoder


def make_decoder(num_topics, num_kmers):
  """Create the decoder function.

  Args:
    num_topics: The number of topics.
    num_kmers: The number of kmers.

  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tfd.Distribution` instance over kmers.
  """
  topics_kmers = tfp.util.TransformedVariable(
      tf.nn.softmax(tf.initializers.GlorotNormal()([num_topics, num_kmers])),
      tfb.SoftmaxCentered(),
      name="topics_kmers")

  def decoder(topics):
    word_probs = tf.matmul(topics, topics_kmers)
    # The observations are bag of kmers and therefore not one-hot. However,
    # log_prob of OneHotCategorical computes the probability correctly in
    # this case.
    return tfd.OneHotCategorical(probs=word_probs, name="bag_of_kmers"), word_probs

  return decoder, topics_kmers


def make_prior(num_topics, initial_value):
  """Create the prior distribution.

  Args:
    num_topics: Number of topics.
    initial_value: The starting value for the prior parameters.

  Returns:
    prior: A `callable` that returns a `tf.distribution.Distribution`
        instance, the prior distribution.
  """
  concentration = tfp.util.TransformedVariable(
      tf.fill([1, num_topics], initial_value),
      tfb.Softplus(),
      name="concentration")

  return tfd.Dirichlet(
      concentration=tfp.util.DeferredTensor(
          concentration, _clip_dirichlet_parameters),
      name="topics_prior")


def model_fn(features, labels, mode, params, config):
  """Build the model function for use in an estimator.

  Args:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.
  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, config

  encoder = make_encoder(params["activation"],
                         params["num_topics"],
                         params["layer_sizes"])
  decoder, topics_kmers = make_decoder(params["num_topics"],
                                       features.shape[1])
  topics_prior = make_prior(params["num_topics"],
                            params["prior_initial_value"])

  alpha = topics_prior.concentration

  topics_posterior = encoder(features)
  topic_posterior_probs = topics_posterior.concentration
  topics = topics_posterior.sample(seed=234)
  random_reconstruction, reconstructed_probs = decoder(topics)

  ## enable this if you want to reconstruct kmers
  #reconstructed_sample = random_reconstruction.sample()   
  if(params['mode'] == "reconstruct"):
    reconstructed_sample = tf1.py_func(
      functools.partial(get_multinomial_sample),
      [reconstructed_probs, features],
      tf.float32,
      stateful=False)

  reconstruction = random_reconstruction.log_prob(features)
  tf1.summary.scalar("reconstruction", tf.reduce_mean(reconstruction))

  # Compute the KL-divergence between two Dirichlets analytically.
  # The sampled KL does not work well for "sparse" distributions
  # (see Appendix D of [2]).
  kl = tfd.kl_divergence(topics_posterior, topics_prior)
  tf1.summary.scalar("kl", tf.reduce_mean(kl))

  # Ensure that the KL is non-negative (up to a very small slack).
  # Negative KL can happen due to numerical instability.
  with tf.control_dependencies(
      [tf.debugging.assert_greater(kl, -1e-3, message="kl")]):
    kl = tf.identity(kl)

  elbo = reconstruction - kl
  avg_elbo = tf.reduce_mean(elbo)
  tf1.summary.scalar("elbo", avg_elbo)
  loss = -avg_elbo

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf1.train.get_or_create_global_step()
  optimizer = tf1.train.AdamOptimizer(params["learning_rate"])

  # This implements the "burn-in" for prior parameters (see Appendix D of [2]).
  # For the first prior_burn_in_steps steps they are fixed, and then trained
  # jointly with the other parameters.
  grads_and_vars = optimizer.compute_gradients(loss)
  grads_and_vars_except_prior = [
      x for x in grads_and_vars if x[1] not in topics_prior.variables]

  def train_op_except_prior():
    return optimizer.apply_gradients(
        grads_and_vars_except_prior,
        global_step=global_step)

  def train_op_all():
    return optimizer.apply_gradients(
        grads_and_vars,
        global_step=global_step)

  train_op = tf.cond(
      pred=global_step < params["prior_burn_in_steps"],
      true_fn=train_op_except_prior,
      false_fn=train_op_all)

  kmers_per_sequence = tf.reduce_sum(features, axis=1)
  log_perplexity = -elbo / kmers_per_sequence
  tf1.summary.scalar("perplexity", tf.exp(tf.reduce_mean(log_perplexity)))
  (log_perplexity_tensor,
   log_perplexity_update) = tf1.metrics.mean(log_perplexity)
  perplexity_tensor = tf.exp(log_perplexity_tensor)

  # Obtain the topics summary. Implemented as a py_func for simplicity.
  topics = tf1.py_func(
      functools.partial(get_topics_strings, vocabulary=params["vocabulary"]),
      [topics_kmers, alpha],
      tf.string,
      stateful=False)
  tf1.summary.text("topics", topics)

  # SAVE BETA to file
  beta_mat = tf1.py_func(
      functools.partial(get_topic_matrix, vocabulary=params["vocabulary"]),
      [topics_kmers, params["prior_initial_value"], params['model_dir']],
      tf.string,
      stateful=False)
  ###########

  if(params['mode'] == "reconstruct"):
    output_matrix = reconstructed_sample
  elif(params['mode'] == "beta"):
    output_matrix = beta_mat
  else:
    output_matrix = topic_posterior_probs

  return tf1.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions=output_matrix, ## topic_posterior_probs, reconstructed_sample, #change this depending on what you want to predict
      eval_metric_ops={
          "elbo": tf1.metrics.mean(elbo),
          "reconstruction": tf1.metrics.mean(reconstruction),
          "kl": tf1.metrics.mean(kl),
          "perplexity": (perplexity_tensor, log_perplexity_update),
          "topics": (topics, tf.no_op()),
      },
  )


def save_topic_posterior(posterior_matrix, outprefix, model_dir):
  posterior_matrix = np.row_stack(posterior_matrix)
  print("[LOG] Type: ",type(posterior_matrix))
  print("[LOG] Shape: ",posterior_matrix.shape)
  outfile = os.path.join(model_dir, format('%s.npy' % outprefix))
  print('[LOG] Saving to: ',outfile)
  np.save(outfile, posterior_matrix)

## reconstruction
def get_multinomial_sample(reconstructed_probs, features):
  reconstructed_sample = []
  kmers_per_sequence = tf.reduce_sum(features, axis=1)
  for doc in range(reconstructed_probs.shape[0]):
    mult = tfd.Multinomial(total_count=kmers_per_sequence[doc],logits=reconstructed_probs[doc,:])
    # fix the seed in the sampling below to keep minimal variance across runs: reconstructed_sample.append(mult.sample(seed=1))
    reconstructed_sample.append(mult.sample())

  return reconstructed_sample


def get_reconstructed_kmers(docs_kmers, vocabulary,
                       topics_to_print=10, kmers_per_sequence=50):
  """Returns reconstructed strings
  Arguments:
    docs_kmers: DxV tensor with documents as rows and kmers as columns.
    vocabulary: A mapping of word's integer index to the corresponding string.
    topics_to_print: The number of top topics to summarize
    kmers_per_topic: Number of kmers per topic to return.
  Returns:
    summary: A np.array with strings.
  """

  docs_kmers = np.row_stack(docs_kmers)
  print(">>>>>>>>>>>> Shape of reconstructed data: ",docs_kmers.shape, 'MAX: ',np.max(docs_kmers))
  res = []
  for doc in range(docs_kmers.shape[0]):
    top_kmers = np.argsort(-docs_kmers[doc,:])
    res.append(" ".join([str(x) for x in top_kmers[0:kmers_per_sequence]]))  ## print ids of top kmers
    l = [docs_kmers[doc, word] for word in top_kmers[0:kmers_per_sequence]]
    res.append(" ".join([str(x) for x in l]))     ## print value of this word in the reconstructed vector
    l = [vocabulary[word] for word in top_kmers[0:kmers_per_sequence]]
    res.append(" ".join(l))

  return np.array(res)


def get_topic_matrix(topics_words, alpha, model_dir, vocabulary):
  """Arguments:
    topics_words: KxV tensor with topics as rows and words as columns.
  Returns:
    summary: A float32
  """
  print("[LOG] Getting the topic matrix: ")
  outfile = os.path.join(model_dir.decode(), format('beta_alpha%g.npy' % alpha))
  res = []
  for topic_idx in range(topics_words.shape[0]):
    l = topics_words[topic_idx,:]
    res.append(l)

  res = np.array(res)
  res = pd.DataFrame(res)
  res.columns = vocabulary
  print('[LOG] Saving to: ',outfile,' size:',res.shape)
  np.save(outfile, res)
  return "placeholder"


def get_topics_strings(topics_kmers, alpha, vocabulary,
                       topics_to_print=10, kmers_per_topic=10):
  """Returns the summary of the learned topics.

  Args:
    topics_kmers: KxV tensor with topics as rows and kmers as columns.
    alpha: 1xK tensor of prior Dirichlet concentrations for the
        topics.
    vocabulary: A mapping of word's integer index to the corresponding string.
    topics_to_print: The number of topics with highest prior weight to
        summarize.
    kmers_per_topic: Number of wodrs per topic to return.
  Returns:
    summary: A np.array with strings.
  """
  alpha = np.squeeze(alpha, axis=0)
  # Use a stable sorting algorithm so that when alpha is fixed
  # we always get the same topics.
  highest_weight_topics = np.argsort(-alpha, kind="mergesort")
  top_kmers = np.argsort(-topics_kmers, axis=1)

  res = []
  for topic_idx in highest_weight_topics[:topics_to_print]:
    l = ["index={} alpha={:.2f}".format(topic_idx, alpha[topic_idx])]
    l += [vocabulary[word] for word in top_kmers[topic_idx, :kmers_per_topic]]
    res.append(" ".join(l))

  return np.array(res)




