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



class HomoTempMERA_RNNCell(RNNCell):
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 grain_width=2,
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
        new_h = Homo_TempMERA_wavefn(inputs,
                                     states,
                                     self.output_size,
                                     self._num_orders,
                                     self._virtual_dim,
                                     self._grain_width)
        new_h = self._activation(new_h)
        return  new_h, new_h




class HomoTempMERA_LSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 grain_width=2,
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
        concat = Homo_TempMERA_wavefn(inputs,
                                      hs,
                                      meta_variable_size,
                                      self._num_orders,
                                      self._virtual_dim,
                                      self._grain_width)
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


#def _list_to_tensor(batch_size, vectors):
#    tensor = vectors[0]
#    for vector in vectors[1:]:
#        tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
#        vector_flat = tf.expand_dims(vector, 1)
#        prod = tf.matmul(tensor_flat, vector_flat)
#        new_shape =  [batch_size] + _shape_value(tensor)[1:] + _shape_value(vector)[1:]
#        tensor = tf.reshape(prod, new_shape)
#    return tensor
def _out_product(batch_size, list_tensor):
    lags = list_tensor.get_shape()[0].value
    tensor = list_tensor[0, :, :]
    for lag in range(1,lags):
        vector = list_tensor[lag, :, :]
        tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
        vector_flat = tf.expand_dims(vector, 1)
        prod = tf.matmul(tensor_flat, vector_flat)
        new_shape =  [batch_size] + _shape_value(tensor)[1:] + _shape_value(vector)[1:]
        tensor = tf.reshape(prod, new_shape)
    return tensor


def _unitary_tensor_shape(_dim):
    return [_dim, _dim, _dim, _dim]


def _isometry_tensor_shape(_low_layer_width, _low_layer_dim, _high_layer_dim):
    shape = [ _low_layer_dim for _ in range(_low_layer_width)]
    shape.append(_high_layer_dim)
    return shape


def homo_bimera_next_layer(old_tensor,
                           disentangler,
                           grainer,
                           grain_width,
                           num_blocks):
    contracted_indx = [[],[]]
    contracted_indx[0] = [ i+2 for i in range(grain_width)]
    contracted_indx[1] = [ i for i in range(grain_width)]
    
    new_tensor = old_tensor
    for _ in range(num_blocks):
        new_tensor = tf.tensordot(new_tensor,
                                  disentangler,
                                  axes = [[1,2], [0,1]])
    for _ in range(num_blocks-1):
        new_tensor = tf.tensordot(new_tensor,
                                  grainer,
                                  axes = contracted_indx)
    
    contracted_indx[0] = [ i+1 for i in range(grain_width)]
    new_tensor = tf.tensordot(new_tensor, grainer, contracted_indx)
    new_tensor = tf.tanh(new_tensor)
    return new_tensor






def Homo_TempMERA_wavefn(inputs,
                         states,
                         meta_size,
                         num_orders,
                         virtual_dim,
                         grain_width=2,
                         bias=True,
                         bias_start=0.0
                         ):
    num_lags = len(states)
    state_size = states[0].get_shape()[1].value         # hidden_size: H
    total_state_size = state_size*num_orders + 1        # [HP + 1]
    
    batch_size = tf.shape(inputs)[0]
    input_size= inputs.get_shape()[1].value     # dimension of variables

    mera_depth, nums_sites = _coarse_graining(num_lags, grain_width)
    
#    states_list = []
#    for state in states:
#        states_list.append(tf.concat([state, tf.ones([batch_size, 1])], 1))
#    state_tensor = _list_to_tensor(batch_size, states_list)
    list_tensors = tf.convert_to_tensor(states)
    first = list_tensors[:,:,:]
    for i in range(2, num_orders+1):
        pow = tf.ones([num_lags, batch_size, state_size]) * i
        powed = tf.pow(first, pow)
        list_tensors = tf.concat([list_tensors, powed], 2)
    list_tensors = tf.concat([list_tensors, tf.ones([num_lags, batch_size, 1])], 2)
    state_tensor = _out_product(batch_size, list_tensors)

    layer = 1
    disentangler = tf.get_variable("disentangler"+str(layer),
                                   shape=_unitary_tensor_shape(total_state_size))
    grainer = tf.get_variable("grainer"+str(layer),
                              shape=_isometry_tensor_shape(grain_width,total_state_size,virtual_dim))
    state_tensor = homo_bimera_next_layer(state_tensor,
                                          disentangler,
                                          grainer,
                                          grain_width,
                                          nums_sites[layer])

    while nums_sites[layer] > 1:
        layer += 1
        disentangler = tf.get_variable("disentangler"+str(layer),
                                       shape=_unitary_tensor_shape(virtual_dim))
        grainer = tf.get_variable("grainer"+str(layer),
                                  shape=_isometry_tensor_shape(grain_width,virtual_dim,virtual_dim))
        state_tensor = homo_bimera_next_layer(state_tensor,
                                              disentangler,
                                              grainer,
                                              grain_width,
                                              nums_sites[layer])
                                              
    weights_h = vs.get_variable("weights_h",
                                [virtual_dim, meta_size],
                                trainable = True)
    weights_x = vs.get_variable("weights_x",
                                [input_size, meta_size],
                                trainable = True)
    out_h = tf.matmul(state_tensor, weights_h)
    out_x = tf.matmul(inputs, weights_x)
    
    res = tf.add(out_x, out_h)
    if not bias:
        return res
    biases = vs.get_variable("biases", [meta_size])
    return nn_ops.bias_add(res,biases)


