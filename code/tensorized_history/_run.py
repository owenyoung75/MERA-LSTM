"""Functions for downloading and reading time series data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base

from reader import read_data_sets
from train_config import *
from models.TempModel_s2s import *
from models.PolyModel_s2s import *
from models.TempHiddenModel_s2s import *

#########################################
"""  Flags for training configuration """
#########################################

flags = tf.flags

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string("model", "TCN", "Model used for learning.")
flags.DEFINE_string("data_path", "./data_genz4.npy", "Data input directory.")
flags.DEFINE_string("save_path", "./log/mps_poly/", "Model output directory.")
flags.DEFINE_bool("use_sched_samp", False, "Use scheduled sampling in training")
flags.DEFINE_integer("inp_steps", 50, "burn in steps")
flags.DEFINE_integer("out_steps", None, "test steps")
flags.DEFINE_integer("batch_size", 50, "batch size")
flags.DEFINE_integer("hidden_size", 4, "hidden layer size")
flags.DEFINE_integer("virtual_dim", 1, "dimension of virtual legs")
flags.DEFINE_integer("num_orders", 1, "order of polynomials")
flags.DEFINE_integer("num_lags", 2, "time-lag length")
flags.DEFINE_integer("num_layers", 2, "time-lag length")
flags.DEFINE_integer("width", 2, "coarse-grain length")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("decay_rate", 0.8, "decay rate")

FLAGS = flags.FLAGS
print('Flags configuration loaded ...')





########################################
"""  Read flags and data into memory """
########################################


# Training Parameters
config = TrainConfig()
config.hidden_size = FLAGS.hidden_size
config.learning_rate = FLAGS.learning_rate
config.decay_rate = FLAGS.decay_rate
config.virtual_dim = FLAGS.virtual_dim
config.virtual_dims = [FLAGS.virtual_dim, FLAGS.virtual_dim, FLAGS.virtual_dim]
config.num_orders = FLAGS.num_orders
config.num_lags = FLAGS.num_lags
config.num_layers = FLAGS.num_layers
config.inp_steps = FLAGS.inp_steps
config.out_steps = FLAGS.out_steps
config.batch_size = FLAGS.batch_size
config.width = FLAGS.width



# Scheduled sampling [optional]
if FLAGS.use_sched_samp:
    config.sample_prob = tf.get_variable("sample_prob",
                                         shape=(),
                                         initializer = tf.zeros_initializer()
                                         )
sampling_burn_in = 400


# Training Parameters
training_steps = config.training_steps
batch_size = config.batch_size
display_step = 500
inp_steps = config.inp_steps
out_steps = config.out_steps




# Read Dataset
dataset, stats = read_data_sets(FLAGS.data_path, True,
                                inp_steps,
                                out_steps,
                                config.num_lags     ## added 07/21/2018
                                )
# Network Parameters
num_input = stats['num_input']  # dataset data input (time series dimension: 1)
num_steps = stats['num_steps']
if out_steps is None:
    out_steps = num_steps - inp_steps



# Print training config
print('-'*130)
print('|input steps|', inp_steps,
      '|out steps|', out_steps,
      '|hidden size|', config.hidden_size,
      '|learning rate|', config.learning_rate,
      '|time lag|', config.num_lags,
      '|poly orders|', config.num_orders,
      '|virtual-D|', config.virtual_dim,
      '|grain width|', config.width
      )
print('-'*130)




###################################
""" Build neural network models """
###################################
#f = open('records_'+str(FLAGS.model)+'.txt', 'w')

Model = globals()[FLAGS.model]

# tf Graph input
X = tf.placeholder("float", [None, inp_steps, num_input])
Y = tf.placeholder("float", [None, out_steps, num_input])
# Decoder output
Z = tf.placeholder("float", [None, out_steps, num_input])


with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=False):
        train_pred = Model(X, Y, True,  config)
with tf.name_scope("Valid"):
    with tf.variable_scope("Model", reuse=True):
        valid_pred = Model(X, Y, False,  config)
with tf.name_scope("Test"):
    with tf.variable_scope("Model", reuse=True):
        test_pred = Model(X, Y, False,  config)

# Define loss and optimizer
train_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(train_pred, Z)))
valid_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(valid_pred, Z)))
test_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(test_pred, Z)))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.001  # Choose an appropriate one.
reg_loss = train_loss + reg_constant * sum(reg_losses)

# Exponential learning rate decay
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = config.learning_rate
learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                           global_step,
                                           2000,
                                           config.decay_rate,
                                           staircase=True)
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_op = optimizer.minimize(train_loss,
                              global_step=global_step)

# Scheduled sampling params
eps_min = 0.1 # minimal prob

## Write summary
#train_summary = tf.summary.scalar('train_loss', train_loss)
#valid_summary = tf.summary.scalar('valid_loss', valid_loss)
#lr_summary = tf.summary.scalar('learning_rate', learning_rate)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

## Saver for the model and loss
saver = tf.train.Saver()
#hist_loss = []

print( str(Model) + 'Model built ...')





###################################
""" Train neural network models """
###################################

with tf.Session() as sess:
    sess.run(init)
    
    training_steps = 2000
    start = time.time()
    for step in range(1, training_steps+1):
        batch_x, batch_y, batch_z = dataset.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op,
                 feed_dict={X: batch_x,
                            Y: batch_y,
                            Z: batch_z}
                 )
        if step % display_step == 0 or step == 1:
            # Calculate batch loss
            tr_loss = sess.run(train_loss,
                               feed_dict={X: batch_x,
                                          Y: batch_y,
                                          Z: batch_z}
                               )
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(tr_loss))
                                            
            # Calculate validation
            valid_enc_inps = dataset.validation.enc_inps.reshape((-1, inp_steps, num_input))
            valid_dec_inps = dataset.validation.dec_inps.reshape((-1, out_steps, num_input))
            valid_dec_outs = dataset.validation.dec_outs.reshape((-1, out_steps, num_input))
            va_loss = sess.run(valid_loss,
                               feed_dict={X: valid_enc_inps,
                                          Y: valid_dec_inps,
                                          Z: valid_dec_outs}
                               )
            print("Validation Loss:", va_loss)
    end = time.time()
    print("Optimization Finished!")
    print("Trainig took: ", end-start, " seconds.")

    # Calculate accuracy for test datasets
    test_enc_inps = dataset.test.enc_inps.reshape((-1, inp_steps, num_input))
    test_dec_inps = dataset.test.dec_inps.reshape((-1, out_steps, num_input))
    test_dec_outs = dataset.test.dec_outs.reshape((-1, out_steps, num_input))
    true, pred, loss = sess.run([Z, test_pred, test_loss],
                                feed_dict={X: test_enc_inps,
                                           Y: test_dec_inps,
                                           Z: test_dec_outs}
                                )
    print("Testing Loss:", loss)



#    # Save the variables to disk.
#    save_path = saver.save(sess, FLAGS.save_path)
#    print("Model saved in file: %s" % save_path)
    # Save predictions
#    np.save("./log/mps_poly/predict.npy", (true, pred))
#    # Save config file
#    with open(save_path+"config.out", 'w') as f:
#        f.write('hidden_size:'+ str(config.hidden_size)+'\n'+
#                'learning_rate:'+ str(config.learning_rate)+ '\n')
#        f.write('train_error:'+ str(loss) +'\n'+
#                'valid_error:' + str(va_loss) + '\n'+
#                'test_error:'+ str(loss) +'\n')
#f.close()

#############################
""" Visualize predictions """
#############################

y_true = true
y_pred = pred

plt.figure
plt.plot(y_true[0,:,0].T,':')
plt.plot(y_pred[0,:,0].T,'-')
plt.savefig("compare.png")
plt.close()

#plt.show()
