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

## local imports
from dataset import build_input_fns, build_fake_input_fns, read_vocabulary
from model import model_fn, save_topic_posterior

tfb = tfp.bijectors
tfd = tfp.distributions


flags.DEFINE_float(
    "learning_rate", default=1e-4, help="Learning rate.")
flags.DEFINE_integer(
    "max_steps", default=300000, help="Number of training steps to run.")
flags.DEFINE_integer(
    "prior_burn_in_steps",
    default=120000,
    help="The number of training steps with fixed prior.")
flags.DEFINE_integer(
    "max_test_steps", default=1, help="Number of test steps to run.")  
flags.DEFINE_integer(
    "test_burn_in_steps",
    default=0,  
    help="The number of test steps with fixed prior.")
flags.DEFINE_integer(
    "num_topics",
    default=100,
    help="The number of topics.")
flags.DEFINE_list(
    "layer_sizes",
    default=["300", "300", "300"],
    help="Comma-separated list denoting hidden units per layer in the encoder.")
flags.DEFINE_string(
    "activation",
    default="relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=16,
    help="Batch size.")
flags.DEFINE_float(
    "prior_initial_value", default=10, help="The initial value for prior.")
flags.DEFINE_string(
    "model_dir",
    default="model_outputs",
    help="Directory to put the model's fit.")
flags.DEFINE_string(
    "vocab_path", default=None,
    help="Path to vocabulary file")  
flags.DEFINE_string(
    "train_path", default=None,
    help="Path to training data file")  
flags.DEFINE_string(
    "eval_path", default=None,
    help="Path to validation data file")  
flags.DEFINE_string(
    "test_path", default=None,
    help="Path to test data file")  
flags.DEFINE_string(
    "preds_file",
    default="test_predictions_selex",
    help="Name of predictions file")  
flags.DEFINE_integer(
    "viz_steps", default=20000, help="Frequency at which save visualizations.")
flags.DEFINE_string(
    "mode",
    default="train",
    help="Mode of algorithm")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing directory.")

FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  print('[LOG] MODE: ',params['mode'])
  params["layer_sizes"] = [int(units) for units in params["layer_sizes"]]
  params["activation"] = getattr(tf.nn, params["activation"])

  params['model_dir'] = os.path.join(params['model_dir'], format('alpha%g_ntopics%d' % (params["prior_initial_value"], params["num_topics"])))

  if(params['mode'] == "test" or params['mode'] == "reconstruct"):
    params["prior_burn_in_steps"] = params["test_burn_in_steps"] 
    params["max_steps"] = params["max_test_steps"]
  elif(params['mode'] == "beta"):
    params["batch_size"] = 1

  print('[LOG] MODEL DIR: ', params['model_dir'])
  tf.io.gfile.makedirs(params['model_dir'])

  print("[LOG] Alpha: ",params["prior_initial_value"], flush=True)
  print("[LOG] Burn-in steps: ",params["prior_burn_in_steps"], flush=True)
  print("[LOG] Num topics: ",params["num_topics"], flush=True)
  print("[LOG] Max steps: ",params["max_steps"], flush=True)
  print("[LOG] Batch-size: ",params["batch_size"], flush=True)


  if(params['mode'] == "train" or params['mode'] == "test"):
    train_input_fn, eval_input_fn, test_input_fn, vocabulary = build_input_fns(params['train_path'], params['eval_path'],
          params['test_path'], params['vocab_path'], params['batch_size'])

  elif(params['mode'] == "beta"):
    vocabulary, ignore = read_vocabulary(params['vocab_path'])
    test_input_fn = build_fake_input_fns(params['batch_size'])

  params["vocabulary"] = vocabulary

  estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=params['model_dir'],
          save_checkpoints_steps=params['viz_steps'],
      ),
  )

  #Fix seed here to minimize variance across runs
  #tf.random.set_seed(1)

  #### print top kmers from learned topics
  if(params['mode'] == "train"):
    for _ in range(params['max_steps'] // params['viz_steps']):
      estimator.train(train_input_fn, steps=params['viz_steps'])
      eval_results = estimator.evaluate(eval_input_fn)
      for key, value in eval_results.items():
        print(key)
        if key == "topics":
          for s in value:
            print(s)
        else:
          print(str(value))
        print("")
      print("")

  print("[LOG] FINISHED TRAINING") 
  if(params['mode'] == "beta"):
    for pred in estimator.predict(test_input_fn):
        break
  else:
      preds = []
      for pred in estimator.predict(test_input_fn):
        preds.append(pred)

  if(params['mode'] == "reconstruct"): ## check reconstruction quality
    kmer_strs = get_reconstructed_kmers(preds, vocabulary)
    outfile = os.path.join(params['model_dir'], format('%s_alpha%g' % (params['preds_file'], params["prior_initial_value"])))
    print('[LOG] Saving to: ',outfile, flush=True)
    np.savetxt(outfile, kmer_strs, fmt='%s')
  elif(params['mode'] == "test" or params['mode'] == "train"):
    save_topic_posterior(preds, format('%s_alpha%g' % (params['preds_file'], params["prior_initial_value"])), params['model_dir'])


if __name__ == "__main__":
  tf1.app.run()
