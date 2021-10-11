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




class TRNNCell(RNNCell):
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 virtual_dims,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 reuse=None
                 ):
        super().__init__(_reuse=reuse)
        self._num_units = num_hidden_units
        self._num_lags = num_lags
        self._virtual_dims = virtual_dims
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
        output = periodic_TT_wavefn(inputs,
                                    states,
                                    self.output_size,
                                    self._virtual_dims,
                                    True)
        new_h = self._activation(output)
        return  new_h, new_h



class TLSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 virtual_dims,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=tanh,
                 reuse=None
                 ):
        super().__init__(_reuse=reuse)
        self._num_units = num_hidden_units
        self._num_lags = num_lags
        self._virtual_dims = virtual_dims
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
        # states: size = time_lag
        if self._state_is_tuple:
            hs = ()
            for state in states:
                c, h = state
                hs += (h,)
        else:
            hs = ()
            for state in states:
                c, h = array_ops.split(value=state,
                                       num_or_size_splits=2,
                                       axis=1)
                hs += (h,)
        
        meta_variable_size = 4 * self.output_size
        concat = periodic_TT_wavefn(inputs,
                                    hs,
                                    meta_variable_size,
                                    self._virtual_dims,
                                    True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
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




def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]

def _outer_product(batch_size, tensor, vector):
    """tensor-vector outer-product"""
    tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
    vector_flat = tf.expand_dims(vector, 1)
    res = tf.matmul(tensor_flat, vector_flat)
    new_shape =  [batch_size]+_shape_value(tensor)[1:]+_shape_value(vector)[1:]
    res = tf.reshape(res, new_shape )
    return res



#def TT_ein_contraction(states_tensor, MPS_tensors):
#    virtual  = "abcdefghijklm"
#    physical = "nopqrstuvwxy"
#
#    def _get_einsum(i, old_legs):
#        A_legs = virtual[i] + physical[i] + virtual[i+1]
#        if i==0:
#            new_legs = old_legs + virtual[i] + virtual[i+1]
#        else:
#            new_legs = old_legs.replace(old_legs[-1], virtual[i+1])
#        new_legs = new_legs.replace(new_legs[1], "")
#        ein = A_legs + "," + old_legs + "->" + new_legs
#        return ein, new_legs
#
#    num_orders = len(MPS_tensors)
#    out_h = states_tensor
#    legs = "z" + physical[:num_orders]
#
#    for i in range(num_orders):
#        einsum, legs = _get_einsum(i, legs)
#        #        print(einsum)
#        out_h = tf.einsum(einsum, MPS_tensors[i], out_h)
#    out_h = tf.squeeze(out_h, [1])
#    return out_h


def TT_contraction(states_tensor, MPS_tensors):
    num_orders = states_tensor.shape.ndims - 1
    
    out_h = tf.tensordot(states_tensor, MPS_tensors[0], [[-1], [1]])
    for i in range(1, num_orders):
        out_h = tf.tensordot(out_h, MPS_tensors[i], [[-3, -1], [1, 0]])
    
    out_h = tf.squeeze(out_h, [1])
    return out_h


def periodic_TT_contraction(states_tensor, MPS_tensors):
    num_orders = len(MPS_tensors) - 1
    
    out_h = tf.tensordot(states_tensor, MPS_tensors[0], [[-1], [1]])
    for i in range(1, num_orders):
        out_h = tf.tensordot(out_h, MPS_tensors[i], [[-3, -1], [1, 0]])

    out_h = tf.tensordot(out_h, MPS_tensors[num_orders], [[1, 2],[2, 0]])
    return out_h






def TT_wavefn(inputs,
              states,
              output_size,
              virtual_dims,
              bias,
              bias_start=0.0
              ):
    num_orders = len(virtual_dims)+1 # alpha_1 to alpha_{K-1}, control the number of copies
    num_lags = len(states)
    state_size = states[0].get_shape()[1].value # hidden_size, i.e. h_{t} dimension
    batch_size = tf.shape(inputs)[0]
    input_size= inputs.get_shape()[1].value     # dimension of variables
    total_state_size = (state_size * num_lags + 1 )     # [HL + 1]
    
    states_vector = tf.concat(states, 1)
    states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1)
    states_tensor = states_vector
    for order in range(num_orders-1):
        states_tensor = _outer_product(batch_size,
                                       states_tensor,
                                       states_vector)


    physical_dims = np.ones((num_orders,)) * total_state_size
    virtual_dims = np.concatenate(([1],
                                  virtual_dims,
                                  [output_size]
                                  ))
    tn_dim = np.cumsum(np.concatenate(([0],
                                       virtual_dims[:-1] * physical_dims * virtual_dims[1:]
                                       )),
                       dtype=np.int32)
    total_dim = tn_dim[-1]
    tsr = vs.get_variable("weights_h",
                          total_dim,
                          trainable = True)
    MPS = []
    for i in range(num_orders):
        mat_A = tf.slice(tsr,
                         [tn_dim[i]],
                         [tn_dim[i + 1] - tn_dim[i]])
        mat_A = tf.reshape(mat_A, [virtual_dims[i],
                                   total_state_size,
                                   virtual_dims[i + 1]])
        MPS.append(mat_A)

    weights_x = vs.get_variable("weights_x",
                                [input_size, output_size],
                                trainable = True)

    out_h = TT_contraction(states_tensor, MPS)
    out_x = tf.matmul(inputs, weights_x)
    res = tf.add(out_x, out_h)

    if not bias:
        return res
    biases = vs.get_variable("biases", [output_size])
    return nn_ops.bias_add(res,biases)


def periodic_TT_wavefn(inputs,
                       states,
                       output_size,
                       virtual_dims,
                       bias,
                       bias_start=0.0
                       ):
    num_orders = len(virtual_dims)
    num_lags = len(states)
    batch_size = inputs.get_shape()[0].value
    input_size= inputs.get_shape()[1].value     # dimension of variables
    state_size = states[0].get_shape()[1].value # hidden_size, i.e. h_{t} dimension
    total_state_size = (state_size * num_lags + 1 )     # [HL + 1]

    """construct augmented state tensor"""
    states_vector = tf.concat(states, 1)    # serialize all h at different time_lags
    states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1) # add the 0th-order: 1
    states_tensor = states_vector
    for order in range(num_orders-1):
        states_tensor = _outer_product(batch_size,
                                       states_tensor,
                                       states_vector)

    physical_dims = np.ones((num_orders,)) * total_state_size
    virtual_dims = np.concatenate((virtual_dims,
                                  [virtual_dims[0]]
                                  ))
    hidden_dim = output_size * virtual_dims[0] * virtual_dims[0]
    tn_dim = np.cumsum(np.concatenate(([0],
                                       virtual_dims[:-1] * physical_dims * virtual_dims[1:]
                                       )),
                       dtype=np.int32)
    total_dim = tn_dim[-1] + hidden_dim
    tsr = vs.get_variable("weights_h",
                          total_dim,
                          trainable = True)
    MPS = []
    for i in range(num_orders):
        mat_A = tf.slice(tsr,
                         [tn_dim[i]],
                         [tn_dim[i + 1] - tn_dim[i]])
        mat_A = tf.reshape(mat_A, [virtual_dims[i],
                                   total_state_size,
                                   virtual_dims[i + 1]])
        MPS.append(mat_A)

    mat_Out = tf.slice(tsr, [tn_dim[num_orders]], [hidden_dim])
    mat_Out = tf.reshape(mat_Out, [virtual_dims[0], output_size, virtual_dims[0]])
    MPS.append(mat_Out)

    weights_x = vs.get_variable("weights_x",
                                [input_size, output_size],
                                trainable = True)

    out_h = periodic_TT_contraction(states_tensor, MPS)
    out_x = tf.matmul(inputs, weights_x)
    res = tf.add(out_x, out_h)

    if not bias:
        return res
    biases = vs.get_variable("biases", [output_size])
    return nn_ops.bias_add(res,biases)





