from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import numpy as np
import copy
from collections import deque




class HomPolyRG_RNNCell(RNNCell):
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 grain_width,
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
        self._grain_width = grain_width
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
        new_h = Homo_PolyRG_wavefn(inputs,
                                   states,
                                   self.output_size,
                                   self._num_orders,
                                   self._virtual_dim,
                                   self._grain_width,
                                   True)
        new_h = self._activation(new_h)
        return  new_h, new_h




class HomoPolyRG_LSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 grain_width,
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
        self._grain_width = grain_width
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
        """this method is inheritated, and always calculate layer by layer"""
        """Now we have multiple states, state->states"""
        sigmoid = tf.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        # states: size = time_lag
        if self._state_is_tuple:
            hs = ()
            for state in states:
                c, h = state    # c and h: tensor_size = (batch_size, hidden_size)
                hs += (h,)      # hs : size = time_lag, i.e. time_lag * (batch_size, hidden_size)
        else:
            hs = ()
            for state in states:
                c, h = array_ops.split(value=state,
                                       num_or_size_splits=2,
                                       axis=1)
                hs += (h,)
        
        meta_variable_size = 4 * self.output_size
        concat = Homo_PolyRG_wavefn(inputs,
                                    hs,
                                    meta_variable_size,
                                    self._num_orders,
                                    self._virtual_dim,
                                    self._grain_width,
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





def _coarse_graining(_num_orders, _grain_width):
    grained_site = _num_orders
    depth = 0
    grained_sites = [grained_site,]
    while grained_site > 1:
        grained_site =  int( (grained_site+_grain_width-1)/_grain_width )
        grained_sites.append(grained_site)
        depth += 1
    contracted_indx = [[],[]]
    for i in range(_grain_width):
        contracted_indx[0].append(i+1)
        contracted_indx[1].append(i)
    return depth, contracted_indx


def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]


def _grained_block(batch_size, _width, state):
    block = state
    for _ in range(_width - 1):
        block_flat= tf.expand_dims(tf.reshape(block, [batch_size,-1]), 2)
        state_flat = tf.expand_dims(state, 1)
        prod = tf.matmul(block_flat, state_flat)
        new_shape =  [batch_size] + _shape_value(block)[1:] + _shape_value(state)[1:]
        block = tf.reshape(prod, new_shape)
    return block


def _rg_tensor_shape(_width, _low_layer_dim, _high_layer_dim):
    shape = []
    for _ in range(_width):
        shape.append(_low_layer_dim)
    shape.append(_high_layer_dim)
    return shape






def Homo_PolyRG_wavefn(inputs,
                       states,
                       meta_size,
                       num_orders,
                       virtual_dim,
                       grain_width,
                       bias,
                       bias_start=0.0
                       ):
    num_lags = len(states)
    state_size = states[0].get_shape()[1].value         # hidden_size, i.e. h_{t} dimension
    total_state_size = (state_size * num_lags + 1 )     # [HL + 1]
    
    batch_size = tf.shape(inputs)[0]
    input_size= inputs.get_shape()[1].value     # dimension of variables


    rg_depth, contracted_indx = _coarse_graining(num_orders, grain_width)
    rg_tensors = []
    if rg_depth == 1:
        rg_tensors.append(tf.get_variable("rgts0",
                                          shape=_rg_tensor_shape(grain_width, total_state_size, virtual_dim)))
    else:
        rg_tensors.append(tf.get_variable("rgts0",
                                          shape=_rg_tensor_shape(grain_width, total_state_size, virtual_dim)))
        for i in range(1, rg_depth):
            rg_tensors.append(tf.get_variable("rgts"+str(i),
                                              shape=_rg_tensor_shape(grain_width, virtual_dim, virtual_dim)))


    states_vector = tf.concat(states, 1)
    states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1)
    grained_state = states_vector
    for i in range(rg_depth):
        grained_block = _grained_block(batch_size,
                                       grain_width,
                                       grained_state)
        grained_state = tf.tensordot(grained_block,
                                     rg_tensors[i],
                                     axes = contracted_indx)
        grained_state = tf.tanh(grained_state)


    weights_h = vs.get_variable("weights_h",
                                [virtual_dim, meta_size],
                                trainable = True)
    weights_x = vs.get_variable("weights_x",
                                [input_size, meta_size],
                                trainable = True)
    out_h = tf.matmul(grained_state, weights_h)
    out_x = tf.matmul(inputs, weights_x)
    
    res = tf.add(out_x, out_h)
    if not bias:
        return res
    biases = vs.get_variable("biases", [meta_size])
    return nn_ops.bias_add(res,biases)


