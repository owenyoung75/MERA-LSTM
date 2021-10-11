from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from tensorflow.python.util import nest
from tensorflow.python.ops.math_ops import tanh

from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.distributions import Bernoulli

import numpy as np
import copy
from collections import deque



class TCN_machine():
    def __init__(self,
                 num_orders,
                 num_lags,
                 virtual_dim,
                 conv_width,
                 reuse=None
                 ):
        self._num_orders = num_orders
        self._num_lags = num_lags
        self._virtual_dim = virtual_dim
        self._conv_width = conv_width

    
    def __call__(self, states):
        num_orders = self._num_orders
        conv_width = self._conv_width
        virtual_dim = self._virtual_dim
        
        num_lags = len(states)
        batch_size = tf.shape(states[0])[0]
        state_size = states[0].get_shape()[1].value
        total_state_size = state_size * num_orders + 1

        multi_sites = tf.transpose(tf.stack(states), perm=[1,0,2])
        
        first = multi_sites[:,:,:]
        for i in range(2, num_orders+1):
            pow = tf.ones([batch_size, num_lags, state_size]) * i
            powed = tf.pow(first, pow)
            multi_sites = tf.concat([multi_sites, powed], 2)
        multi_sites = tf.concat([multi_sites, tf.ones([batch_size, num_lags, 1])], 2)

        depth = 0
        filter = tf.get_variable("filter"+str(depth),
                                 shape=[conv_width, total_state_size, virtual_dim])
        multi_sites = tf.nn.conv1d(multi_sites,
                                   filters = filter,
                                   stride = conv_width,
                                   padding = "SAME")
        multi_sites = tf.nn.pool(multi_sites,
                                 window_shape = [conv_width],
                                 pooling_type = "AVG",
                                 padding = "SAME",
                                 strides = [conv_width])
                                 
        while multi_sites.shape[1] > 1:
            depth += 1
            filter = tf.get_variable("filter"+str(depth),
                                     shape=[conv_width, virtual_dim, virtual_dim])
            multi_sites = tf.nn.conv1d(multi_sites,
                                       filters = filter,
                                       stride = conv_width,
                                       padding = "SAME")
            multi_sites = tf.nn.pool(multi_sites,
                                     window_shape = [conv_width],
                                     pooling_type = "AVG",
                                     padding = "SAME",
                                     strides = [conv_width])
        multi_sites = tf.squeeze(multi_sites, [1])
        weights = vs.get_variable("weights",
                                  [virtual_dim, state_size],
                                  trainable = True)
        res = tf.matmul(multi_sites, weights)
        biases = vs.get_variable("biases", [state_size])
        res = nn_ops.bias_add(res,biases)
        return  res


