from __future__ import print_function

import tensorflow as tf

from .PolyRep.MPS import *
from .PolyRep.MERA import *
from .PolyRep.MERA_homo import *
from .PolyRep.PCN import *
from .PolyRep.RG import *
from .PolyRep.RG_homo import *


from .PolyRep._poly_model_imply import *



def PolyMPS(enc_inps,
            dec_inps,
            is_training,
            config):
    print('Training -->') if is_training else print('Testing -->')
    machine = PolyMPS_machine(config.num_lags,
                              config.virtual_dims)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = poly_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = poly_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs






def PolyMERA(enc_inps,
             dec_inps,
             is_training,
             config):
    print('Training -->') if is_training else print('Testing -->')
    machine = PolyMERA_machine(config.num_lags,
                               config.num_orders,
                               config.virtual_dim,
                               config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = poly_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = poly_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs



def HomoPolyMERA(enc_inps,
                 dec_inps,
                 is_training,
                 config):
    print('Training -->') if is_training else print('Testing -->')
    machine = PolyMERA_machine(config.num_lags,
                               config.num_orders,
                               config.virtual_dim,
                               config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = poly_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = poly_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs




def PolyCNN(enc_inps,
            dec_inps,
            is_training,
            config):
    print('Training -->') if is_training else print('Testing -->')
    machine = PCN_machine(config.num_lags,
                          config.num_orders,
                          config.virtual_dim,
                          config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = poly_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = poly_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs




def PolyRG(enc_inps,
           dec_inps,
           is_training,
           config):
    print('Training -->') if is_training else print('Testing -->')
    machine = PolyRG_machine(config.num_lags,
                             config.num_orders,
                             config.virtual_dim,
                             config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = poly_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = poly_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs



def HomoPolyRG(enc_inps,
               dec_inps,
               is_training,
               config):
    print('Training -->') if is_training else print('Testing -->')
    machine = HomoPolyRG_machine(config.num_lags,
                                 config.num_orders,
                                 config.virtual_dim,
                                 config.width)
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs = poly_tn_with_feed_prev(machine,
                                          enc_inps,
                                          True,
                                          config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        dec_outs = poly_tn_with_feed_prev(machine,
                                          dec_inps,
                                          is_training,
                                          config)
    return dec_outs

