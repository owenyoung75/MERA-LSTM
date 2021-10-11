from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import numpy as np
import copy
from collections import deque



def poly_tn_with_feed_prev(machine,
                           inputs,
                           is_training,
                           config
                           ):
    output_states = []

    with tf.variable_scope("poly") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0] 
        num_steps = inputs_shape[1]         ###    12 for training(enc_inp) and 88 for testing(dec_inp)
        inp_steps =  config.inp_steps
        
        states_list = []
        for step in range(config.num_lags):
            states_list.append(inputs[:, step, :])

        for time_step in range(num_steps):
#        for time_step in range(3):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            
            if is_training or time_step < inp_steps:
                states_list[-1] = inputs[:, time_step, :]

            output_state = machine(states_list)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output_state = tf.sigmoid(output_state)
                output_states.append(output_state)
            
            states_list.append(output_state)
            states_list.pop(0)
    
    output_states = tf.stack(output_states, 1)
    return output_states



