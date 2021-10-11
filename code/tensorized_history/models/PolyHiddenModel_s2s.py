from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from models.PolyHiddenRep.TT import *
from models.PolyHiddenRep.PCN import *
from models.PolyHiddenRep.MPS import *
from models.PolyHiddenRep.MPS_homo import *
from models.PolyHiddenRep.RG import *
from models.PolyHiddenRep.RG_homo import *
from models.PolyHiddenRep.MERA import *
from models.PolyHiddenRep.MERA_homo import *
#from PolyRep.PolyCV import *
#from PolyRep.PolyCV_homo import *

from models.PolyHiddenRep._poly_trnn_imply import *



def RNN(enc_inps, dec_inps,is_training, config):
    def rnn_cell():
        return rnn.BasicRNNCell(config.hidden_size)
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(rnn_cell(), output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([rnn_cell() for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = rnn_with_feed_prev(cell,
                                                  enc_inps,
                                                  True,
                                                  config)
    with tf.variable_scope("Decoder", reuse=None):
        config.inp_steps = 0
        dec_outs, dec_states = rnn_with_feed_prev(cell,
                                                  dec_inps,
                                                  is_training,
                                                  config,
                                                  enc_states)
    return dec_outs


def LSTM(enc_inps, dec_inps, is_training, config):
    def lstm_cell():
        return rnn.BasicLSTMCell(config.hidden_size,
                                 forget_bias=1.0,
                                 reuse=None)
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([lstm_cell() for _ in range(config.num_layers)])

    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = rnn_with_feed_prev(cell,
                                                  enc_inps,
                                                  True, config)
    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = rnn_with_feed_prev(cell,
                                                  dec_inps,
                                                  is_training,
                                                  config,
                                                  enc_states)






def TRNN(enc_inps,
         dec_inps,
         is_training,
         config):
    def trnn_cell():
        return TRNNCell(config.hidden_size,
                        config.num_lags,
                        config.virtual_dims)
    print('Training -->') if is_training else print('Testing -->')
    cell= trnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs


def TLSTM(enc_inps,
          dec_inps,
          is_training,
          config):
    def tlstm_cell():
        return TLSTMCell(config.hidden_size,
                         config.num_lags,
                         config.virtual_dims)
    print('Training -->') if is_training else print('Testing -->')
    cell= tlstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs






def PolyMPS_RNN(enc_inps,
                dec_inps,
                is_training,
                config):
    def mps_rnn_cell():
        return PolyMPS_RNNCell(config.hidden_size,
                               config.num_lags,
                               config.virtual_dims)
    print('Training -->') if is_training else print('Testing -->')
    cell= mps_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs


def PolyMPS_LSTM(enc_inps,
                 dec_inps,
                 is_training,
                 config):
    def mps_lstm_cell():
        return PolyMPS_LSTMCell(config.hidden_size,
                                config.num_lags,
                                config.virtual_dims)
    print('Training -->') if is_training else print('Testing -->')
    cell= mps_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs






def PolyMERA_RNN(enc_inps,
                 dec_inps,
                 is_training,
                 config):
    def mera_rnn_cell():
        return PolyMERA_RNNCell(config.hidden_size,
                                config.num_lags,
                                config.num_orders,
                                config.virtual_dim)
    print('Training -->') if is_training else print('Testing -->')
    cell= mera_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs


def PolyMERA_LSTM(enc_inps,
                  dec_inps,
                  is_training,
                  config):
    def mera_lstm_cell():
        return PolyMERA_LSTMCell(config.hidden_size,
                                 config.num_lags,
                                 config.num_orders,
                                 config.virtual_dim)
    print('Training -->') if is_training else print('Testing -->')
    cell= mera_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs







def PolyRG_RNN(enc_inps,
               dec_inps,
               is_training,
               config):
    def polyrg_rnn_cell():
        return PolyRG_RNNCell(config.hidden_size,
                              config.num_lags,
                              config.num_orders,
                              config.virtual_dim,
                              config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= polyrg_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs


def PolyRG_LSTM(enc_inps,
                dec_inps,
                is_training,
                config):
    def polyrg_lstm_cell():
        return PolyRG_LSTMCell(config.hidden_size,
                               config.num_lags,
                               config.num_orders,
                               config.virtual_dim,
                               config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= polyrg_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs








#def PolyCV_RNN(enc_inps,
#               dec_inps,
#               is_training,
#               config):
#    def polycv_rnn_cell():
#        return PolyCV_RNNCell(config.hidden_size,
#                              config.num_lags,
#                              config.num_orders,
#                              config.virtual_dim,
#                              config.width)
#    print('Training -->') if is_training else print('Testing -->')
#    cell= polycv_rnn_cell()
#    if is_training and config.keep_prob < 1:
#        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
#    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
#    with tf.variable_scope("Encoder", reuse=None):
#        print(' '*10+'Create Encoder ...')
#        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
#                                                         enc_inps,
#                                                         True,
#                                                         config)
#    with tf.variable_scope("Decoder", reuse=None):
#        print(' '*10+'Create Decoder ...')
#        config.inp_steps = 0
#        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
#                                                         dec_inps,
#                                                         is_training,
#                                                         config,
#                                                         enc_states)
#    return dec_outs
#
#
#def PolyCV_LSTM(enc_inps,
#                dec_inps,
#                is_training,
#                config):
#    def polycv_lstm_cell():
#        return PolyCV_LSTMCell(config.hidden_size,
#                               config.num_lags,
#                               config.num_orders,
#                               config.virtual_dim,
#                               config.width)
#    print('Training -->') if is_training else print('Testing -->')
#    cell= polycv_lstm_cell()
#    if is_training and config.keep_prob < 1:
#        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
#    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
#    with tf.variable_scope("Encoder", reuse=None):
#        print(' '*10+'Create Encoder ...')
#        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
#                                                         enc_inps,
#                                                         True,
#                                                         config)
#    with tf.variable_scope("Decoder", reuse=None):
#        print(' '*10+'Create Decoder ...')
#        config.inp_steps = 0
#        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
#                                                         dec_inps,
#                                                         is_training,
#                                                         config,
#                                                         enc_states)
#    return dec_outs







def PolyCNN_RNN(enc_inps,
                dec_inps,
                is_training,
                config):
    def pcn_rnn_cell():
        return PCN_RNNCell(config.hidden_size,
                           config.num_lags,
                           config.num_orders,
                           config.virtual_dim,
                           config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= pcn_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs


def PolyCNN_LSTM(enc_inps,
                 dec_inps,
                 is_training,
                 config):
    def pcn_lstm_cell():
        return PCN_LSTMCell(config.hidden_size,
                            config.num_lags,
                            config.num_orders,
                            config.virtual_dim,
                            config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= pcn_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs






def HomoPolyMPS_RNN(enc_inps,
                    dec_inps,
                    is_training,
                    config):
    def homomps_rnn_cell():
        return HomoMPS_RNNCell(config.hidden_size,
                               config.num_lags,
                               config.num_orders,
                               config.virtual_dim)
    print('Training -->') if is_training else print('Testing -->')
    cell= homomps_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs


def HomoPolyMPS_LSTM(enc_inps,
                     dec_inps,
                     is_training,
                     config):
    def homomps_lstm_cell():
        return HomoMPS_LSTMCell(config.hidden_size,
                                config.num_lags,
                                config.num_orders,
                                config.virtual_dim)
    print('Training -->') if is_training else print('Testing -->')
    cell= homomps_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs





#def HomoPolyCV_RNN(enc_inps,
#                   dec_inps,
#                   is_training,
#                   config):
#    def homopolycv_rnn_cell():
#        return HomoPolyCV_RNNCell(config.hidden_size,
#                                  config.num_lags,
#                                  config.num_orders,
#                                  config.virtual_dim,
#                                  config.width)
#    print('Training -->') if is_training else print('Testing -->')
#    cell= homopolycv_rnn_cell()
#    if is_training and config.keep_prob < 1:
#        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
#    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
#    with tf.variable_scope("Encoder", reuse=None):
#        print(' '*10+'Create Encoder ...')
#        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
#                                                         enc_inps,
#                                                         True,
#                                                         config)
#    with tf.variable_scope("Decoder", reuse=None):
#        print(' '*10+'Create Decoder ...')
#        config.inp_steps = 0
#        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
#                                                         dec_inps,
#                                                         is_training,
#                                                         config,
#                                                         enc_states)
#    return dec_outs
#
#
#def HomoPolyCV_LSTM(enc_inps,
#                    dec_inps,
#                    is_training,
#                    config):
#    def homopolycv_lstm_cell():
#        return HomoPolyCV_LSTMCell(config.hidden_size,
#                                   config.num_lags,
#                                   config.num_orders,
#                                   config.virtual_dim,
#                                   config.width)
#    print('Training -->') if is_training else print('Testing -->')
#    cell= homopolycv_lstm_cell()
#    if is_training and config.keep_prob < 1:
#        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
#    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
#    with tf.variable_scope("Encoder", reuse=None):
#        print(' '*10+'Create Encoder ...')
#        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
#                                                         enc_inps,
#                                                         True,
#                                                         config)
#    with tf.variable_scope("Decoder", reuse=None):
#        print(' '*10+'Create Decoder ...')
#        config.inp_steps = 0
#        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
#                                                         dec_inps,
#                                                         is_training,
#                                                         config,
#                                                         enc_states)
#    return dec_outs







def HomoPolyRG_RNN(enc_inps,
                   dec_inps,
                   is_training,
                   config):
    def homopolyrg_rnn_cell():
        return HomoPolyRG_RNNCell(config.hidden_size,
                                  config.num_lags,
                                  config.num_orders,
                                  config.virtual_dim,
                                  config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= homopolyrg_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs


def HomoPolyRG_LSTM(enc_inps,
                    dec_inps,
                    is_training,
                    config):
    def homopolyrg_lstm_cell():
        return HomoPolyRG_LSTMCell(config.hidden_size,
                                   config.num_lags,
                                   config.num_orders,
                                   config.virtual_dim,
                                   config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= homopolyrg_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs








def HomoPolyMERA_RNN(enc_inps,
                     dec_inps,
                     is_training,
                     config):
    def homomera_rnn_cell():
        return HomoPolyMERA_RNNCell(config.hidden_size,
                                    config.num_lags,
                                    config.num_orders,
                                    config.virtual_dim,
                                    config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= homomera_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs


def HomoPolyMERA_LSTM(enc_inps,
                      dec_inps,
                      is_training,
                      config):
    def homomera_lstm_cell():
        return HomoPolyMERA_LSTMCell(config.hidden_size,
                                     config.num_lags,
                                     config.num_orders,
                                     config.virtual_dim,
                                     config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= homomera_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell,
                                                         enc_inps,
                                                         True,
                                                         config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell,
                                                         dec_inps,
                                                         is_training,
                                                         config,
                                                         enc_states)
    return dec_outs
