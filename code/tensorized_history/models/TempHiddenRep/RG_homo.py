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



class HomoTempRG_RNNCell(RNNCell):
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
        new_h = Homo_TempRG_wavefn(inputs,
                                   states,
                                   self.output_size,
                                   self._num_orders,
                                   self._virtual_dim,
                                   self._grain_width,
                                   True)
        new_h = self._activation(new_h)
        return  new_h, new_h




class HomoTempRG_LSTMCell(RNNCell):
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
        concat = Homo_TempRG_wavefn(inputs,
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





def _coarse_graining(_num_sites, _width):
    grained_num = _num_sites
    nums_sites = [grained_num,]
    while grained_num > 1:
        grained_num =  int( (grained_num+_width-1)/_width )
        nums_sites.append(grained_num)
    depth = len(nums_sites)-1
    return depth, nums_sites


def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]


def _block_tensor(batch_size, _grain_width, vectors):
    tensor = vectors[0]
    for vector in vectors[1:]:
        tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
        vector_flat = tf.expand_dims(vector, 1)
        prod = tf.matmul(tensor_flat, vector_flat)
        new_shape =  [batch_size] + _shape_value(tensor)[1:] + _shape_value(vector)[1:]
        tensor = tf.reshape(prod, new_shape)
    return tensor


def _rg_tensor_shape(_width, _low_layer_dim, _high_layer_dim):
    shape = [_low_layer_dim for _ in range(_width)]
    shape.append(_high_layer_dim)
    return shape


def grained_lattice(old_lattice,
                    rgtr,
                    grain_width,
                    num_blocks):
    contracted_indx = [[],[]]
    for i in range(grain_width):
        contracted_indx[0].append(i+1)
        contracted_indx[1].append(i)

    new_lattice = []
    for i in range(num_blocks):
        if (i+1)*grain_width > len(old_lattice):
            block = old_lattice[i*grain_width:]
            block += old_lattice[ : ((i+1)*grain_width - len(old_lattice))]
        else:
            block = old_lattice[i*grain_width : (i+1)*grain_width]
        block_tensor = _block_tensor(tf.shape(old_lattice[0])[0],
                                     grain_width,
                                     block)
        new_site = tf.tensordot(block_tensor,
                                rgtr,
                                axes = contracted_indx)
        new_site = tf.tanh(new_site)
        new_lattice.append(new_site)
    return new_lattice



def Homo_TempRG_wavefn(inputs,
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
    total_state_size = state_size*num_orders + 1        # [HP + 1]

    
    batch_size = tf.shape(inputs)[0]
    input_size= inputs.get_shape()[1].value     # dimension of variables


    rg_depth, num_sites = _coarse_graining(num_lags, grain_width)
    rg_tensors = []
    if rg_depth == 1:
        rg_tensors.append(tf.get_variable("rgts1",
                                          shape=_rg_tensor_shape(grain_width,
                                                                 total_state_size,
                                                                 virtual_dim)))
    else:
        rg_tensors.append(tf.get_variable("rgts1",
                                          shape=_rg_tensor_shape(grain_width,
                                                                 total_state_size,
                                                                 virtual_dim)))
        for i in range(1, rg_depth):
            rg_tensors.append(tf.get_variable("rgts"+str(i+1),
                                              shape=_rg_tensor_shape(grain_width,
                                                                     virtual_dim,
                                                                     virtual_dim)))

#    lattice = []
#    for state in states:
#        lattice.append(tf.concat([state, tf.ones([batch_size, 1])], 1))
    list_tensors = tf.stack(states)
    first = list_tensors[:,:,:]
    for i in range(2, num_orders+1):
        pow = tf.ones([num_lags, batch_size, state_size]) * i
        powed = tf.pow(first, pow)
        list_tensors = tf.concat([list_tensors, powed], 2)
    list_tensors = tf.concat([list_tensors, tf.ones([num_lags, batch_size, 1])], 2)
    lattice = tf.unstack(list_tensors)


    for layer in range(rg_depth):
        lattice = grained_lattice(lattice,
                                  rg_tensors[layer],
                                  grain_width,
                                  num_sites[layer+1])


    weights_h = vs.get_variable("weights_h",
                                [virtual_dim, meta_size],
                                trainable = True)
    weights_x = vs.get_variable("weights_x",
                                [input_size, meta_size],
                                trainable = True)
    out_h = tf.matmul(lattice[0], weights_h)
    out_x = tf.matmul(inputs, weights_x)
    
    res = tf.add(out_x, out_h)
    if not bias:
        return res
    biases = vs.get_variable("biases", [meta_size])
    return nn_ops.bias_add(res,biases)


