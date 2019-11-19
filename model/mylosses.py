import tensorflow as tf
from keras import losses
from keras import backend as K
import numpy as np

from model import lovasz_losses_tf

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_ls01_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.1) + dice_loss(y_true, y_pred)
    return loss

def bce_ls02_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2) + dice_loss(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def lovasz_hinge_loss(y_true, y_pred):
    eps = 1e-5
    logit_pred = K.log(y_pred + eps) - K.log(1.0 - y_pred + eps)
    loss = lovasz_losses_tf.lovasz_hinge(logit_pred, y_true, per_image=True, ignore=None)
    return loss

def mask_bce_ls01_dice_loss(y_true, y_pred):
    """
    y_true : shape=(batch, h, w, class)
    y_pred : shape=(batch, h, w, class+1), last class is all of y_true
    """
    mask_true = tf.reduce_max(y_true, axis=-1, keepdims=True)
    y_true_add_mask = tf.concat([y_true, mask_true], axis=-1)
    loss = bce_ls01_dice_loss(y_true_add_mask, y_pred)
    return loss

def l1_margin01_dice_loss(y_true, y_pred):
    MARGIN = 0.1
    l1_loss = K.relu(K.abs(y_true - y_pred) - MARGIN)
    loss = l1_loss + dice_loss(y_true, K.relu(y_pred))
    return loss

def l1_margin005_dice_loss(y_true, y_pred):
    MARGIN = 0.05
    l1_loss = K.relu(K.abs(y_true - y_pred) - MARGIN)
    loss = l1_loss + dice_loss(y_true, K.relu(y_pred))
    return loss

def l1_margin00_dice_loss(y_true, y_pred):
    MARGIN = 0.0
    l1_loss = K.relu(K.abs(y_true - y_pred) - MARGIN)
    loss = l1_loss + dice_loss(y_true, K.relu(y_pred))
    return loss

def l1_margin01(y_true, y_pred):
    MARGIN = 0.1
    l1_loss = K.relu(K.abs(y_true - y_pred) - MARGIN)
    loss = l1_loss
    return loss

def my_l_n_margin01_dice_loss_wrapper(c0, c1):
    def my_l_n_margin01_dice_loss(y_true, y_pred):
        MARGIN = 0.1
        n = (1 - y_true) * (c0 - c1) + c1
        l_n = tf.pow(K.relu(K.abs(y_true - y_pred) - MARGIN), n)
        loss = l_n + dice_loss(y_true, K.relu(y_pred))
        return loss
    return my_l_n_margin01_dice_loss

def cce_dice_loss(y_true, y_pred):

    #n_class = K.int_shape(y_pred)[-1]
    #s = np.sqrt(2) * np.log(n_class - 1)
    #s = 2

    #logit = s * y_pred
    #sm = tf.nn.softmax(logit, axis=-1)
    #x_sm = tf.nn.relu(y_pred) * sm

    #x_exp = tf.nn.relu(y_pred) * tf.exp(logit)
    #x_sm = x_exp / (tf.reduce_sum(x_exp, axis=-1, keepdims=True) + 1e-7)

    #x = tf.nn.relu(y_pred)
    #x_sm = x / (tf.reduce_sum(x, axis=-1, keepdims=True) + 1e-7)


    #label = y_true / (tf.reduce_sum(y_true, axis=-1, keepdims=True) + 1e-7)

    #cce = - tf.reduce_sum(label * tf.log(x_sm + 1e-7), axis=-1)
    #cce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit, axis=-1)
    #loss = cce + dice_loss(y_true, K.relu(y_pred))
    loss = dice_loss(y_true, K.relu(y_pred))

    return loss
