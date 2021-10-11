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




class PolyMPS_machine():
    def __init__(self,
                 num_lags,
                 virtual_dims,
                 reuse = None
                 ):
        self._num_lags = num_lags
        self._virtual_dims = virtual_dims
    
    def __call__(self, states):
        num_orders = len(self._virtual_dims)
        num_lags = len(states)
        batch_size = tf.shape(states[0])[0]
        state_size = states[0].get_shape()[1].value
        total_state_size = (state_size * num_lags + 1 )     # [ML + 1]
    
        """construct augmented state tensor"""
        states_vector = tf.concat(states, 1)
        states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1)
    
        physical_dims = np.ones((num_orders,)) * total_state_size
        virtual_dims  = np.concatenate((self._virtual_dims,
                                        [self._virtual_dims[0]]
                                        ))
        output_dim = state_size * virtual_dims[0] * virtual_dims[0]
        tn_dim = np.cumsum(np.concatenate(([0],
                                           virtual_dims[:-1] * physical_dims * virtual_dims[1:]
                                           )),
                           dtype=np.int32)
        total_dim = tn_dim[-1] + output_dim
        
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        tsr = vs.get_variable(name = "weights",
                              shape = total_dim,
                              regularizer = regularizer,
                              trainable = True)
        Matrices = []
        for i in range(num_orders):
            mat_A = tf.slice(tsr,
                             [tn_dim[i]],
                             [tn_dim[i + 1] - tn_dim[i]])
            mat_A = tf.reshape(mat_A, [virtual_dims[i],
                                       total_state_size,
                                       virtual_dims[i + 1]])
            Matrices.append(mat_A)

        mat_Out = tf.slice(tsr, [tn_dim[num_orders]], [output_dim])
        mat_Out = tf.reshape(mat_Out, [virtual_dims[0], state_size, virtual_dims[0]])
        Matrices.append(mat_Out)

        res = periodic_matrix_contraction(states_vector, Matrices)

        bias_start=0.0
        biases = vs.get_variable("biases", [state_size])
        res = nn_ops.bias_add(res,biases)
        return res



def periodic_matrix_contraction(states_vector, MPS_tensors):
    num_orders = len(MPS_tensors) - 1
    for i in range(num_orders):
        MPS_tensors[i] = tf.tensordot(states_vector, MPS_tensors[i], [[1], [1]])
    
    out = MPS_tensors[0]
    for i in range(1, num_orders):
        out = tf.matmul(out, MPS_tensors[i])

    out = tf.tensordot(out, MPS_tensors[-1], [[1, 2],[2, 0]])
    return out






