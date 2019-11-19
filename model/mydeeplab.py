from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, LeakyReLU, Lambda
from keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Dropout
from keras.layers import Concatenate, Add, Multiply
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import numpy as np

from model import deeplab_v3
from model.cos_similarity import CosSimilarityWithFeatvec, ActivityRegularization_L2NormToConst


def mydeeplab_v1(input_shape, num_class):
    DOWNSIZE_RATE = 1/2
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))
    
    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.output

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)

    #
    model = Model(input = inputs, output = oup)

    return model

def mydeeplab_v2(input_shape, num_class):
    DOWNSIZE_RATE = 1/2
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))
    
    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3(cityscapes)
    dplb_model = deeplab_v3.Deeplabv3(weights='cityscapes', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.output

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)

    #
    model = Model(input = inputs, output = oup)

    return model

def mydeeplab_mask_v1(input_shape, num_class):
    DOWNSIZE_RATE = 1/2
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))
    
    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class+1, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.output

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    x = Activation('sigmoid')(x)
    
    mask = Lambda(lambda _x: _x[...,-2:-1])(x)
    temp_prob = Lambda(lambda _x: _x[...,:-1])(x)    
    prob = Multiply()([temp_prob, mask])

    oup = Concatenate()([prob, mask])

    #
    model = Model(input = inputs, output = oup)

    return model

def mydeeplab_v3(input_shape, num_class):
    DOWNSIZE_RATE = 2/3
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))
    
    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.output

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)

    #
    model = Model(input = inputs, output = oup)

    return model

def mydeeplab_v4(input_shape, num_class):
    DOWNSIZE_RATE = 4/5
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))
    
    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.output

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)

    #
    model = Model(input = inputs, output = oup)

    return model

def mydeeplab_v5(input_shape, num_class):
    DOWNSIZE_RATE = 1
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))
    
    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.output

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)

    #
    model = Model(input = inputs, output = oup)

    return model

def mydeeplab_mask_v2(input_shape, num_class):
    DOWNSIZE_RATE = 2/3
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))
    
    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class+1, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.output

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    x = Activation('sigmoid')(x)
    
    mask = Lambda(lambda _x: _x[...,-2:-1])(x)
    temp_prob = Lambda(lambda _x: _x[...,:-1])(x)    
    prob = Multiply()([temp_prob, mask])

    oup = Concatenate()([prob, mask])

    #
    model = Model(input = inputs, output = oup)

    return model

def mydeeplab_l2norm_v1(input_shape, num_class, scale):
    DOWNSIZE_RATE = 2/3
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))
    
    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    #x = dplb_model.output
    x = dplb_model.layers[161].output # feature map
    #x = dplb_model.layers[162].output # dropout feature map

    # L2 normalization
    x = Lambda(lambda _x: tf.math.l2_normalize(_x, axis=-1), name='L2_normalization')(x)
    x = Lambda(lambda _x: _x * scale, name='scaling')(x)

    # custom_logits_semantic
    x = Conv2D(num_class, (1, 1), padding='same', name='custom_logits_semantic')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)

    #
    model = Model(input = inputs, output = oup)

    return model


def mydeeplab_featvec_v1(input_shape, num_class, regu_coef=None, act_regu_coef=None, oup_act='relu', downsize_rate=2/3):
    DOWNSIZE_RATE = downsize_rate
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))

    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.layers[161].output # feature map (B, H, W, F)

    # cosine similarity (B, H, W, Class)
    if act_regu_coef is not None:
        n_ele = K.int_shape(x)[-1]
        x = ActivityRegularization_L2NormToConst(c=np.sqrt(n_ele), coef=act_regu_coef, axis=-1)(x)
    x = CosSimilarityWithFeatvec(n_vec=num_class, regu_coef=regu_coef, name='cos_similarity_with_featvec')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    if oup_act=='leakyrelu':
        oup = LeakyReLU(alpha=0.1)(x)
    else:
        oup = Activation(oup_act)(x)

    #
    model = Model(input = inputs, output = oup)

    return model
def mydeeplab_featvec_v2(input_shape, num_class, regu_coef=None, act_regu_coef=None, oup_act='relu', downsize_rate=2/3):
    DOWNSIZE_RATE = downsize_rate
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))

    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.layers[160].output # feature map (batch normalization before relu) (B, H, W, F)
    #x = dplb_model.layers[161].output # feature map (relu) (B, H, W, F)

    # cosine similarity (B, H, W, Class)
    if act_regu_coef is not None:
        n_ele = K.int_shape(x)[-1]
        x = ActivityRegularization_L2NormToConst(c=np.sqrt(n_ele), coef=act_regu_coef, axis=-1)(x)
    x = CosSimilarityWithFeatvec(n_vec=num_class, regu_coef=regu_coef, name='cos_similarity_with_featvec')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    if oup_act=='leakyrelu':
        oup = LeakyReLU(alpha=0.1)(x)
    else:
        oup = Activation(oup_act)(x)

    #
    model = Model(input = inputs, output = oup)

    return model
def mydeeplab_featvec_v2_2(input_shape, num_class, regu_coef=None, act_regu_coef=None, oup_act='relu', scaling=False, downsize_rate=2/3):
    DOWNSIZE_RATE = downsize_rate
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))

    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.layers[159].output # feature map (conv before batchnormalization) (B, H, W, F)
    #x = dplb_model.layers[160].output # feature map (batchnormalization before relu) (B, H, W, F)
    #x = dplb_model.layers[161].output # feature map (relu) (B, H, W, F)

    # scaling
    if scaling:
        x = BatchNormalization(center=False)(x)

    # cosine similarity (B, H, W, Class)
    if act_regu_coef is not None:
        n_ele = K.int_shape(x)[-1]
        x = ActivityRegularization_L2NormToConst(c=np.sqrt(n_ele), coef=act_regu_coef, axis=-1)(x)
    x = CosSimilarityWithFeatvec(n_vec=num_class, regu_coef=regu_coef, name='cos_similarity_with_featvec')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    if oup_act=='leakyrelu':
        oup = LeakyReLU(alpha=0.1)(x)
    else:
        oup = Activation(oup_act)(x)

    #
    model = Model(input = inputs, output = oup)

    return model

def mydeeplab_featvec_v3(input_shape, num_class, n_vec, downsize_rate=2/3):
    DOWNSIZE_RATE = downsize_rate
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))

    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.layers[160].output # feature map (batch normalization before relu) (B, H, W, F)
    #x = dplb_model.layers[161].output # feature map (relu) (B, H, W, F)

    # cosine similarity (B, H, W, Class)
    x = CosSimilarityWithFeatvec(n_vec=n_vec, name='cos_similarity_with_featvec')(x)

    # custom_logits_semantic
    x = Conv2D(num_class, (1, 1), padding='same', name='custom_logits_semantic')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)
    #oup = Activation('relu')(x)

    #
    model = Model(input = inputs, output = oup)

    return model
def mydeeplab_featvec_v3_1(input_shape, num_class, n_vec, downsize_rate=2/3):
    DOWNSIZE_RATE = downsize_rate
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))

    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.layers[160].output # feature map (batch normalization before relu) (B, H, W, F)
    #x = dplb_model.layers[161].output # feature map (relu) (B, H, W, F)

    # cosine similarity (B, H, W, Class)
    x = CosSimilarityWithFeatvec(n_vec=n_vec, name='cos_similarity_with_featvec')(x)
    x = Activation('relu')(x)

    # custom_logits_semantic
    x = Conv2D(num_class, (1, 1), padding='same', name='custom_logits_semantic')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)
    #oup = Activation('relu')(x)

    #
    model = Model(input = inputs, output = oup)

    return model
def mydeeplab_featvec_v4(input_shape, num_class, n_vec, n_last_hidden, downsize_rate=2/3):
    DOWNSIZE_RATE = downsize_rate
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))

    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.layers[160].output # feature map (batch normalization before relu) (B, H, W, F)
    #x = dplb_model.layers[161].output # feature map (relu) (B, H, W, F)

    # cosine similarity (B, H, W, Class)
    x = CosSimilarityWithFeatvec(n_vec=n_vec, name='cos_similarity_with_featvec')(x)

    # hidden
    x = Conv2D(n_last_hidden, (1, 1), padding='same', name='last_hidden', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    # custom_logits_semantic
    x = Conv2D(num_class, (1, 1), padding='same', name='custom_logits_semantic')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)
    #oup = Activation('relu')(x)

    #
    model = Model(input = inputs, output = oup)

    return model
def mydeeplab_featvec_v4_1(input_shape, num_class, n_vec, n_last_hidden, downsize_rate=2/3):
    DOWNSIZE_RATE = downsize_rate
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))

    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.layers[160].output # feature map (batch normalization before relu) (B, H, W, F)
    #x = dplb_model.layers[161].output # feature map (relu) (B, H, W, F)

    # cosine similarity (B, H, W, Class)
    x = CosSimilarityWithFeatvec(n_vec=n_vec, name='cos_similarity_with_featvec')(x)

    # hidden
    x = Conv2D(n_last_hidden, (1, 1), padding='same', name='last_hidden', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # custom_logits_semantic
    x = Conv2D(num_class, (1, 1), padding='same', name='custom_logits_semantic')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)
    #oup = Activation('relu')(x)

    #
    model = Model(input = inputs, output = oup)

    return model
def mydeeplab_featvec_v4_2(input_shape, num_class, n_vec, n_last_hidden, downsize_rate=2/3):
    DOWNSIZE_RATE = downsize_rate
    downsize = (int(input_shape[0] * DOWNSIZE_RATE), int(input_shape[1] * DOWNSIZE_RATE))

    # input
    inputs = Input(input_shape)
    x = inputs

    # downsize
    x = Lambda(lambda _x: tf.image.resize(_x, downsize))(x) #BILINEAR

    # Deeplabv3
    dplb_model = deeplab_v3.Deeplabv3(weights='pascal_voc', input_tensor=x, input_shape=K.int_shape(x)[1:], classes=num_class, backbone='mobilenetv2', OS=16, alpha=1., activation=None)
    x = dplb_model.layers[160].output # feature map (batch normalization before relu) (B, H, W, F)
    #x = dplb_model.layers[161].output # feature map (relu) (B, H, W, F)

    # cosine similarity (B, H, W, Class)
    x = CosSimilarityWithFeatvec(n_vec=n_vec, name='cos_similarity_with_featvec')(x)
    x = Activation('relu')(x)

    # hidden
    x = Conv2D(n_last_hidden, (1, 1), padding='same', name='last_hidden', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)

    # custom_logits_semantic
    x = Conv2D(num_class, (1, 1), padding='same', name='custom_logits_semantic')(x)

    # upsampling
    x = Lambda(lambda _x: tf.image.resize(_x, input_shape[:2]))(x) #BILINEAR

    # output
    oup = Activation('sigmoid')(x)
    #oup = Activation('relu')(x)

    #
    model = Model(input = inputs, output = oup)

    return model


# wrapper
def mydeeplab_featvec_wrapper_relu_last(deeplab_featvec_model):
    inp = deeplab_featvec_model.input
    x = deeplab_featvec_model.output
    oup = Activation('relu')(x)
    model = Model(input=inp, output=oup)
    return model