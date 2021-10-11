from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from models.TempHiddenRep.TCN import *
from models.TempHiddenRep.MERA import *
from models.TempHiddenRep.MERA_homo import *
from models.TempHiddenRep.RG import *
from models.TempHiddenRep.RG_homo import *

from models.TempHiddenRep._temp_trnn_imply import *





def TempMERA_RNN(enc_inps,
                 dec_inps,
                 is_training,
                 config):
    def mera_rnn_cell():
        return TempMERA_RNNCell(config.hidden_size,
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
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs


def TempMERA_LSTM(enc_inps,
                  dec_inps,
                  is_training,
                  config):
    def mera_lstm_cell():
        return TempMERA_LSTMCell(config.hidden_size,
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
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs







def TempRG_RNN(enc_inps,
               dec_inps,
               is_training,
               config):
    def temprg_rnn_cell():
        return TempRG_RNNCell(config.hidden_size,
                              config.num_lags,
                              config.num_orders,
                              config.virtual_dim,
                              config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= temprg_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs


def TempRG_LSTM(enc_inps,
                dec_inps,
                is_training,
                config):
    def temprg_lstm_cell():
        return TempRG_LSTMCell(config.hidden_size,
                               config.num_lags,
                               config.num_orders,
                               config.virtual_dim,
                               config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= temprg_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs









def TempCNN_RNN(enc_inps,
                dec_inps,
                is_training,
                config):
    def tcn_rnn_cell():
        return TCN_RNNCell(config.hidden_size,
                           config.num_lags,
                           config.num_orders,
                           config.virtual_dim,
                           config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= tcn_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs


def TempCNN_LSTM(enc_inps,
                 dec_inps,
                 is_training,
                 config):
    def tcn_lstm_cell():
        return TCN_LSTMCell(config.hidden_size,
                            config.num_lags,
                            config.num_orders,
                            config.virtual_dim,
                            config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= tcn_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs







def HomoTempRG_RNN(enc_inps,
                   dec_inps,
                   is_training,
                   config):
    def homotemprg_rnn_cell():
        return HomoTempRG_RNNCell(config.hidden_size,
                                  config.num_lags,
                                  config.num_orders,
                                  config.virtual_dim,
                                  config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= homotemprg_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs


def HomoTempRG_LSTM(enc_inps,
                    dec_inps,
                    is_training,
                    config):
    def homotemprg_lstm_cell():
        return HomoTempRG_LSTMCell(config.hidden_size,
                                   config.num_lags,
                                   config.num_orders,
                                   config.virtual_dim,
                                   config.width)
    print('Training -->') if is_training else print('Testing -->')
    cell= homotemprg_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs







def HomoTempMERA_RNN(enc_inps,
                     dec_inps,
                     is_training,
                     config):
    def homomera_rnn_cell():
        return HomoTempMERA_RNNCell(config.hidden_size,
                                    config.num_lags,
                                    config.num_orders,
                                    config.virtual_dim)
    print('Training -->') if is_training else print('Testing -->')
    cell= homomera_rnn_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs


def HomoTempMERA_LSTM(enc_inps,
                      dec_inps,
                      is_training,
                      config):
    def homomera_lstm_cell():
        return HomoTempMERA_LSTMCell(config.hidden_size,
                                     config.num_lags,
                                     config.num_orders,
                                     config.virtual_dim)
    print('Training -->') if is_training else print('Testing -->')
    cell= homomera_lstm_cell()
    if is_training and config.keep_prob < 1:
        cell = rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
    cell = rnn.MultiRNNCell([cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = temp_rnn_with_feed_prev(cell,
                                                       enc_inps,
                                                       True,
                                                       config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = temp_rnn_with_feed_prev(cell,
                                                       dec_inps,
                                                       is_training,
                                                       config,
                                                       enc_states)
    return dec_outs
