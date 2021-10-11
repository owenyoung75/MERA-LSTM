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



class TCN_RNNCell(RNNCell):
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 conv_width,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 reuse=None
                 ):
        super().__init__(_reuse=reuse)
        self._num_units = num_hidden_units
        self._num_lags = num_lags
        self._virtual_dim = virtual_dim
        self._conv_width = conv_width
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units
    
    def __call__(self, inputs, states):
        """this method is inheritated, and always calculate layer by layer"""
        new_h = TCN_wavefn(inputs,
                           states,
                           self.output_size,
                           self._num_orders,
                           self._virtual_dim,
                           self._conv_width,
                           True)
        new_h = self._activation(new_h)
        return  new_h, new_h




class TCN_LSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 conv_width,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 reuse=None
                 ):
        super().__init__(_reuse=reuse)
        self._num_units = num_hidden_units
        self._num_lags = num_lags
        self._num_orders = num_orders
        self._virtual_dim = virtual_dim
        self._conv_width = conv_width
        self._forget_bias = forget_bias
        self._state_is_tuple= state_is_tuple
        self._activation = activation
        
    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)
    
    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, states):
        sigmoid = tf.sigmoid
        if self._state_is_tuple:    # states: size = time_lag
            hs = ()
            for state in states:
                c, h = state        # c and h: tensor_size = (batch_size, hidden_size)
                hs += (h,)          # hs : size = time_lag, i.e. time_lag * (batch_size, hidden_size)
        else:
            hs = ()
            for state in states:
                c, h = array_ops.split(value=state,
                                       num_or_size_splits=2,
                                       axis=1)
                hs += (h,)
        
        meta_variable_size = 4 * self.output_size
        concat = TCN_wavefn(inputs,
                            hs,
                            meta_variable_size,
                            self._num_orders,
                            self._virtual_dim,
                            self._conv_width,
                            True)
        i, j, f, o = array_ops.split(value=concat,
                                     num_or_size_splits=4,
                                     axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state









def TCN_wavefn(inputs,
               states,
               meta_size,
               num_orders,
               virtual_dim,
               conv_width,
               bias=True,
               bias_start=0.0
               ):               
    num_lags = len(states)
    state_size = states[0].get_shape()[1].value         # hidden_size, i.e. h_{t} dimension
    total_state_size = state_size * num_orders + 1      # [HP + 1]

    batch_size = tf.shape(inputs)[0]
    input_size= inputs.get_shape()[1].value     # dimension of variables

#    states_list = []
#    for state in states:
#        states_list.append(tf.concat([state, tf.ones([batch_size, 1])], 1))
#    multi_sites = tf.transpose(tf.convert_to_tensor(states_list), perm=[1,0,2])
    multi_sites = tf.transpose(tf.convert_to_tensor(states), perm=[1,0,2])
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


    weights_h = vs.get_variable("weights_h",
                                [virtual_dim, meta_size],
                                trainable = True)
    weights_x = vs.get_variable("weights_x",
                                [input_size, meta_size],
                                trainable = True)
    out_h = tf.matmul(multi_sites, weights_h)
    out_x = tf.matmul(inputs, weights_x)
    
    res = tf.add(out_x, out_h)
    if not bias:
        return res
    biases = vs.get_variable("biases", [meta_size])
    return nn_ops.bias_add(res,biases)


