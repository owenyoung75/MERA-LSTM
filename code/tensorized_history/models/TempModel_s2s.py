from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from .TempRep.TCN import *
from .TempRep.MERA import *
from .TempRep.MERA_homo import *
from .TempRep.RG import *
from .TempRep.RG_homo import *

from .TempRep._temp_model_imply import *



def TempMERA(enc_inps,
             dec_inps,
             is_training,
             config):
    print('Training -->') if is_training else print('Testing -->')
    machine = TempMERA_machine(config.num_orders,
                               config.num_lags,
                               config.virtual_dim,
                               config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = temp_tn_with_feed_prev(machine,
                                                      enc_inps,
                                                      True,
                                                      config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = temp_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs



def HomoTempMERA(enc_inps,
             dec_inps,
             is_training,
             config):
    print('Training -->') if is_training else print('Testing -->')
    machine = HomoTempMERA_machine(config.num_orders,
                                   config.num_lags,
                                   config.virtual_dim,
                                   config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = temp_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = temp_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs




def TempCNN(enc_inps,
            dec_inps,
            is_training,
            config):
    print('Training -->') if is_training else print('Testing -->')
    machine = TCN_machine(config.num_orders,
                          config.num_lags,
                          config.virtual_dim,
                          config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = temp_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = temp_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs





def TempRG(enc_inps,
           dec_inps,
           is_training,
           config):
    print('Training -->') if is_training else print('Testing -->')
    machine = TempRG_machine(config.num_orders,
                             config.num_lags,
                             config.virtual_dim,
                             config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = temp_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = temp_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs



def HomoTempRG(enc_inps,
               dec_inps,
               is_training,
               config):
    print('Training -->') if is_training else print('Testing -->')
    machine = HomoTempRG_machine(config.num_orders,
                                 config.num_lags,
                                 config.virtual_dim,
                                 config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = temp_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = temp_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs


