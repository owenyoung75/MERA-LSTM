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




class PolyMERA_machine():
    def __init__(self,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 grain_width=2,
                 reuse=None
                 ):
        self._num_lags = num_lags
        self._num_orders = num_orders
        self._virtual_dim = virtual_dim
        self._grain_width = grain_width
    
    def __call__(self, states):
        grain_width=self._grain_width
        bias=True
        bias_start=0.0
        num_orders = self._num_orders
        virtual_dim = self._virtual_dim
        num_lags = len(states)
        state_size = states[0].get_shape()[1].value         # dimension of variables
        total_state_size = (state_size * num_lags + 1 )     # [ML + 1]
    
        batch_size = tf.shape(states[0])[0]
    
        mera_depth, nums_sites = _coarse_graining(num_orders, grain_width)
    
        states_vector = tf.concat(states, 1)
        states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1)
        lattice = [states_vector for _ in range(num_orders)]
        state_tensor = _list_to_tensor(batch_size, grain_width, lattice)
    
        layer = 1; disentg_tensors = []; grain_tensors = []
        for i in range(nums_sites[layer]):
            disentg_tensors.append(tf.get_variable("disentangler"+str(layer)+str(i),
                                                   shape=_unitary_tensor_shape(total_state_size)))
            grain_tensors.append(tf.get_variable("grainer"+str(layer)+str(i),
                                                 shape=_isometry_tensor_shape(grain_width,
                                                                              total_state_size,
                                                                              virtual_dim)))
        state_tensor = bimera_next_layer(state_tensor,
                                         disentg_tensors,
                                         grain_tensors,
                                         grain_width)
        while nums_sites[layer] > 1:
            layer += 1; disentg_tensors = []; grain_tensors = []
            for i in range(nums_sites[layer]):
                disentg_tensors.append(tf.get_variable("disentangler"+str(layer)+str(i),
                                                       shape=_unitary_tensor_shape(virtual_dim)))
                grain_tensors.append(tf.get_variable("grainer"+str(layer)+str(i),
                                                     shape=_isometry_tensor_shape(grain_width,
                                                                                  virtual_dim,
                                                                                  virtual_dim)))
            state_tensor = bimera_next_layer(state_tensor,
                                             disentg_tensors,
                                             grain_tensors,
                                             grain_width)

        weights = vs.get_variable("weights",
                                  [virtual_dim, state_size],
                                  trainable = True)
        res = tf.matmul(state_tensor, weights)
        biases = vs.get_variable("biases", [state_size])
        res = nn_ops.bias_add(res,biases)
        return res








def _coarse_graining(_num_orders, _width):
    grained_num = _num_orders
    nums_sites = [grained_num,]
    while grained_num > 1:
        grained_num =  int( (grained_num+_width-1)/_width )
        nums_sites.append(grained_num)
    depth = len(nums_sites)-1
    return depth, nums_sites


def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]


def _list_to_tensor(batch_size, _grain_width, vectors):
    tensor = vectors[0]
    for vector in vectors[1:]:
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


def bimera_next_layer(old_tensor,
                      disentg_tensors,
                      grain_tensors,
                      grain_width):
    contracted_indx = [[],[]]
    contracted_indx[0] = [ i+2 for i in range(grain_width)]
    contracted_indx[1] = [ i for i in range(grain_width)]

    new_tensor = old_tensor
    for disentangler in disentg_tensors:
        new_tensor = tf.tensordot(new_tensor,
                                  disentangler,
                                  axes = [[1,2], [0,1]])
    for grainer in grain_tensors[1:]:
        new_tensor = tf.tensordot(new_tensor,
                                  grainer,
                                  axes = contracted_indx)

    contracted_indx[0] = [ i+1 for i in range(grain_width)]
    new_tensor = tf.tensordot(new_tensor, grain_tensors[0], contracted_indx)
    return new_tensor



