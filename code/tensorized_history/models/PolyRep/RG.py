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




class PolyRG_machine():
    def __init__(self,
                 num_lags,
                 num_orders,
                 virtual_dim,
                 grain_width,
                 reuse=None
                 ):
        self._num_lags = num_lags
        self._num_orders = num_orders
        self._virtual_dim = virtual_dim
        self._grain_width = grain_width

    
    def __call__(self, states):
        
        num_lags = len(states)
        virtual_dim = self._virtual_dim
        grain_width = self._grain_width
        num_orders = self._num_orders
        state_size = states[0].get_shape()[1].value         # hidden_size, i.e. h_{t} dimension
        total_state_size = (state_size * num_lags + 1 )     # [ML + 1]
        batch_size = tf.shape(states[0])[0]
    
        rg_depth, nums_sites = _coarse_graining(num_orders, grain_width)
    
        states_vector = tf.concat(states, 1)
        states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1)
        lattice = [states_vector for _ in range(num_orders)]
        
        layer = 1; layer_tensors = []
        for i in range(nums_sites[layer]):
            layer_tensors.append(tf.get_variable("rgts"+str(layer)+str(i),
                                                 shape=_rg_tensor_shape(grain_width,
                                                                        total_state_size,
                                                                        virtual_dim)))
        lattice = grained_lattice(lattice,
                                  layer_tensors,
                                  grain_width,
                                  nums_sites[layer])
                                  
        while nums_sites[layer] > 1:
            layer += 1; layer_tensors = []
            for i in range(nums_sites[layer]):
                layer_tensors.append(tf.get_variable("rgts"+str(layer)+str(i),
                                                     shape=_rg_tensor_shape(grain_width,
                                                                            virtual_dim,
                                                                            virtual_dim)))
            lattice = grained_lattice(lattice,
                                      layer_tensors,
                                      grain_width,
                                      nums_sites[layer])
                
        weights = vs.get_variable("weights",
                                  [virtual_dim, state_size],
                                  trainable = True)

        res = tf.matmul(lattice[0], weights)
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


def _block_tensor(batch_size, _grain_width, vectors):
    tensor = vectors[0]
    for vector in vectors[1:]:
        tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
        vector_flat = tf.expand_dims(vector, 1)
        prod = tf.matmul(tensor_flat, vector_flat)
        new_shape =  [batch_size] + _shape_value(tensor)[1:] + _shape_value(vector)[1:]
        tensor = tf.reshape(prod, new_shape)
    return tensor


def _rg_tensor_shape(_low_layer_width, _low_layer_dim, _high_layer_dim):
    shape = []
    for _ in range(_low_layer_width):
        shape.append(_low_layer_dim)
    shape.append(_high_layer_dim)
    return shape


def grained_lattice(old_lattice,
                    rgts,
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
                                rgts[i],
                                axes = contracted_indx)
        new_site = tf.tanh(new_site)
        new_lattice.append(new_site)
                                
    return new_lattice


