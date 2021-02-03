from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sys import platform
from datetime import datetime

from gcn.utils import *
from gcn.models import GCN, MLP

import pandas as pd
pd.options.mode.chained_assignment = None  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'default', 'Dataset string.')  # 'delta'
flags.DEFINE_string('seasons', '2020', 'Seasons to include.')  # '2014 - 2020, comma separated'
flags.DEFINE_string('train', 'LPL', 'Leauge(s) to train on')  
flags.DEFINE_string('val', 'LCK', 'Leauge(s) to validate on')  
flags.DEFINE_string('test', 'LCS', 'Leauge(s) to test on')  
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')

# Load data
if platform == 'darwin':
  data_root = 'data/'
else:
  data_root = '..\\..\\deep-learning-class-project\\data\\'
seasons = FLAGS.seasons.split(',')
data_paths = []
for year in seasons:
  data_paths.append('{}{}_LoL_esports_match_data_from_OraclesElixir_20201005.csv'.format(data_root,year))


train_league = ['LPL']
valid_league = ['LCK']
test_leagues = ['LCS' ,'LEC','VCS','PCS']
test_out = []

#######################################
# Begin CV
#######################################
for layer_count in [16]:
  FLAGS.hidden1 = layer_count
  FLAGS.hidden2 = layer_count
  for dropout in [0.5]:
    FLAGS.dropout = dropout
    for test_league in test_leagues:
      data_division = (train_league,valid_league,[test_league])
      adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, data_cols = load_data_lol(data_paths,FLAGS.dataset,data_division)

      # Some preprocessing
      features = preprocess_features(features)

      if FLAGS.model == 'gcn':
          support = [preprocess_adj(adj)]
          num_supports = 1
          model_func = GCN
      elif FLAGS.model == 'gcn_cheby':
          support = chebyshev_polynomials(adj, FLAGS.max_degree)
          num_supports = 1 + FLAGS.max_degree
          model_func = GCN
      elif FLAGS.model == 'dense':
          support = [preprocess_adj(adj)]  # Not used
          num_supports = 1
          model_func = MLP
      else:
          raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

      # Define placeholders
      placeholders = {
          'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
          'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
          'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
          'labels_mask': tf.placeholder(tf.int32),
          'dropout': tf.placeholder_with_default(0., shape=()),
          'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
      }

      # Create model
      model = model_func(placeholders, input_dim=features[2][1], logging=True)

      # Initialize session
      sess = tf.Session()

      # add logging
      log_dir = "logs/".format(FLAGS.train,FLAGS.val,FLAGS.test,FLAGS.seasons)

      # Define model evaluation function
      def evaluate(features, support, labels, mask, placeholders):
          t_test = time.time()
          feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
          outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
          return outs_val[0], outs_val[1], (time.time() - t_test)


      # Init variables
      sess.run(tf.global_variables_initializer())

      cost_val = []

      # Train model
      print('\nIn training loop...')
      for epoch in range(FLAGS.epochs):

          t = time.time()
          # Construct feed dictionary
          feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
          feed_dict.update({placeholders['dropout']: FLAGS.dropout})

          # Training step
          outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

          # Validation
          cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
          cost_val.append(cost)

          if epoch % 100 == 0:
            print('Epoch : {}, train acc : {:.3f}, val acc : {:.3f}'.format(epoch, outs[2], acc))

          if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
              print("Stopped at early at epoch {}".format(epoch))
              break

      print("Optimization Finished!")
      test_eval = evaluate(features, support, y_test, test_mask, placeholders)
      test_res = [test_league, FLAGS.model, FLAGS.hidden1, FLAGS.dropout, round(test_eval[0],5), round(test_eval[1],5)]
      test_out.append(test_res)
      print('Test on {}, {}, {}, {}: loss "{:.5f}", acc "{:.5f}".'.format(*test_res))
      sess.close()

test_frame = pd.DataFrame(
  columns = ['league','model','layer_count','dropout','cost','acc'],
  data = test_out
)
print(test_frame)
test_frame.to_csv(log_dir + 'test_league_scores_gcn_cheby_degree_1.csv',index=False)
