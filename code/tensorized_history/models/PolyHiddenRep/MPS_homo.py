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




class HomoPolyMPS_RNNCell(RNNCell):
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
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
        new_h = Homo_PolyMPS_wavefn(inputs,
                                    states,
                                    self.output_size,
                                    self._num_orders,
                                    self._virtual_dim,
                                    True)
        new_h = self._activation(new_h)
        return  new_h, new_h




class HomoPolyMPS_LSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self,
                 num_hidden_units,
                 num_lags,
                 num_orders,
                 virtual_dim,
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
        concat = Homo_PolyMPS_wavefn(inputs,
                                     hs,
                                     meta_variable_size,
                                     self._num_orders,
                                     self._virtual_dim,
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






def homo_periodic_contraction(states_state, MPS_tensors, num_orders):
    MPS_tensors[0] = tf.tensordot(states_state, MPS_tensors[0], [[1], [1]])
    
    out_h = MPS_tensors[0]
    for _ in range(num_orders-1):
        out_h = tf.matmul(out_h, MPS_tensors[0])

    out_h = tf.tensordot(out_h, MPS_tensors[1], [[1, 2],[2, 0]])
    return out_h



def Homo_PolyMPS_wavefn(inputs,
                        states,
                        output_size,
                        num_orders,
                        virtual_dim,
                        bias,
                        bias_start=0.0
                        ):
    num_lags = len(states)
    state_size = states[0].get_shape()[1].value         # hidden_size, i.e. h_{t} dimension
    total_state_size = (state_size * num_lags + 1 )     # [HL + 1]
    
    batch_size = tf.shape(inputs)[0]
    input_size= inputs.get_shape()[1].value     # dimension of variables

    """construct augmented state tensor"""
    states_vector = tf.concat(states, 1)
    states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1)


    """ construct a big serialized variable for all MPS parameters """
    tsA_size   = total_state_size * virtual_dim * virtual_dim
    tsOut_size = output_size * virtual_dim * virtual_dim
    tsr = vs.get_variable("weights_h",
                          tsA_size + tsOut_size,
                          trainable = True)
    
    ts_A = tf.slice(tsr, [0], [tsA_size] )
    ts_A = tf.reshape(ts_A, [virtual_dim,
                             total_state_size,
                             virtual_dim])
    ts_Out = tf.slice(tsr, [tsA_size], [tsOut_size] )
    ts_Out = tf.reshape(ts_Out, [virtual_dim,
                                 output_size,
                                 virtual_dim])
                                 
    MPS = [ts_A, ts_Out]

    weights_x = vs.get_variable("weights_x",
                                [input_size, output_size],
                                trainable = True)

    out_h = homo_periodic_contraction(states_vector, MPS, num_orders)
    
    out_x = tf.matmul(inputs, weights_x)
    
    res = tf.add(out_x, out_h)
    if not bias:
        return res
    biases = vs.get_variable("biases", [output_size])
    return nn_ops.bias_add(res,biases)


