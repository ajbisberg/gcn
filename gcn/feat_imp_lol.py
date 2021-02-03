from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sys import platform
from datetime import datetime
from scipy.sparse import csr_matrix,lil_matrix
from eli5.permutation_importance import get_score_importances
import pandas as pd

from gcn.utils import *
from gcn.models import GCN, MLP

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
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')



# Load data
if platform == 'darwin':
  data_root = 'data/'
else:
  data_root = '..\\..\\deep-learning-class-project\\data\\'
seasons = FLAGS.seasons.split(',')
data_paths = []
for year in seasons:
  data_paths.append('{}{}_LoL_esports_match_data_from_OraclesElixir_20201005.csv'.format(data_root,year))
data_division = (FLAGS.train.split(','),FLAGS.val.split(','),FLAGS.test.split(','))
adj, raw_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, data_cols = load_data_lol(data_paths,FLAGS.dataset,data_division)

# Some preprocessing
features = preprocess_features(raw_features)

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
log_dir = "logs/{}-{}-{}-{}".format(FLAGS.train,FLAGS.val,FLAGS.test,FLAGS.seasons)
# train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
# val_writer = tf.summary.FileWriter(log_dir + '/val')
# with tf.name_scope('performance-train{}_val{}_test{}_seasons{}'.format(FLAGS.train,FLAGS.val,FLAGS.test,FLAGS.seasons)):
#     # Summaries need to be displayed
#     # Whenever you need to record the loss, feed the mean loss to this placeholder
#     tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
#     # Create a scalar summary object for the loss so it can be displayed
#     tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

#     # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
#     tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
#     # Create a scalar summary object for the accuracy so it can be displayed
#     tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

# performance_summaries = tf.summary.merge([tf_loss_summary,tf_accuracy_summary])

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

def score_train(X,y_train):
    features = preprocess_features(csr_matrix(X))
    feed_dict_val = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    _, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return acc

def score_val(X,y_val):
    features = preprocess_features(csr_matrix(X))
    feed_dict_val = construct_feed_dict(features, support, y_val, val_mask, placeholders)
    _, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return acc

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
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

    # Print results
    print("Epoch:", '%04d' % (epoch + 1),
        #"train_loss=", "{:.5f}".format(outs[1]),
        "train_acc=", "{:.5f}".format(outs[2]), 
        #"val_loss=", "{:.5f}".format(cost),
        "val_acc=", "{:.5f}".format(acc), 
        "time=", "{:.5f}".format(time.time() - t))
    
    # train_summ = sess.run(performance_summaries, feed_dict={tf_loss_ph:outs[1], tf_accuracy_ph:outs[2]})
    # train_writer.add_summary(train_summ, epoch)
    # val_summ = sess.run(performance_summaries, feed_dict={tf_loss_ph:cost, tf_accuracy_ph:acc})
    # val_writer.add_summary(val_summ, epoch)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

print("Running train feature importance ...")
base_score, score_decreases = get_score_importances(score_train, raw_features.toarray(), y_train)
mean_feat_imp = np.mean(score_decreases, axis=0)
std_feat_imp = np.std(score_decreases, axis=0)
feat_imp_stats = pd.DataFrame(
  columns = data_cols,
  data = [mean_feat_imp, std_feat_imp]
)
feat_imp_stats.to_csv(log_dir + '/train/feat_imp.csv',index=False)

print("Running validation feature importance ...")
base_score, score_decreases = get_score_importances(score_val, raw_features.toarray(), y_val)
mean_feat_imp = np.mean(score_decreases, axis=0)
std_feat_imp = np.std(score_decreases, axis=0)
feat_imp_stats = pd.DataFrame(
  columns = data_cols,
  data = [mean_feat_imp, std_feat_imp]
)
feat_imp_stats.to_csv(log_dir + '/val/feat_imp.csv',index=False)