from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda
from keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers import Concatenate, Add
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf

from model.drop_activation import DropActivation

def _conv_bn_relu(filters, kernel_size, strides=1, activation='relu', l2reg=1e-4, dilation_rate=1, drop_activation=None):
    def f(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding='same',
                   dilation_rate=dilation_rate,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2reg)
                   )(x)
        x = BatchNormalization()(x)
        if activation is not None:
            if drop_activation is not None:
                x = DropActivation(rate=drop_activation)(x)
            else:
                x = Activation(activation)(x)
        return x
    return f

def _maxpooling_conv(filters, kernel_size, n_conv=2, activation='relu', last_activation=None, drop_activation=None, l2reg=1e-4, dilation_rate=1, maxpooling=True, shortcut=None):
    def f(x):
        # maxpooling -> conv
        x1 = x
        if maxpooling:
            x1 = MaxPooling2D(pool_size=(2, 2))(x1)
        for i in range(n_conv):
            if i < n_conv-1:
                x1 = _conv_bn_relu(filters, kernel_size, activation=activation, drop_activation=drop_activation, l2reg=l2reg, dilation_rate=dilation_rate)(x1)
            else:
                x1 = _conv_bn_relu(filters, kernel_size, activation=last_activation, drop_activation=drop_activation, l2reg=l2reg, dilation_rate=dilation_rate)(x1)

        # shortcut
        if shortcut is None:
            y = x1
        else:
            x2 = x
            if maxpooling:
                if shortcut == 'ave':
                    x2 = AveragePooling2D(pool_size=(2, 2))(x2)
                else:
                    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
            
            if K.int_shape(x2)[-1] != filters:
                x2 = Conv2D(filters=filters, kernel_size=(1, 1), strides=1, kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4))(x2)

            y = Add()([x1, x2])
        return y
    return f

def _upsampling_conv_concate_conv(filters, kernel_size, n_conv=2, activation='relu', last_activation=None, drop_activation=None, l2reg=1e-4):
    def f(xs):
        x1 = xs[0] # upsampling
        x2 = xs[1] # shortcut

        x1 = UpSampling2D(size=(2,2))(x1)
        x1 = _conv_bn_relu(filters, kernel_size, activation=activation, drop_activation=drop_activation, l2reg=l2reg)(x1)

        x = Concatenate(axis=3)([x2, x1])
        
        for i in range(n_conv):
            if i < n_conv-1:
                x = _conv_bn_relu(filters, kernel_size, activation=activation, drop_activation=drop_activation, l2reg=l2reg)(x)
            else:
                x = _conv_bn_relu(filters, kernel_size, activation=last_activation, drop_activation=drop_activation, l2reg=l2reg)(x)
        return x
    return f



def unet_v1(input_shape, num_class):
    INITIAL_FILTER = 64
    INITIAL_KERNEL_SIZE = 7
    INITIAL_STRIDE = 2


    # 1
    inputs = Input(input_shape)
    x = inputs

    # 1/S
    conv1 = _conv_bn_relu(filters=INITIAL_FILTER, kernel_size=INITIAL_KERNEL_SIZE, strides=INITIAL_STRIDE, activation='relu')(x)
    conv1 = _maxpooling_conv(INITIAL_FILTER, 3, n_conv=2, activation='relu', last_activation=None, maxpooling=False)(conv1)
    # 1/S * 1/2
    conv2 = _maxpooling_conv(INITIAL_FILTER*2, 3, n_conv=2, activation='relu', last_activation=None)(conv1)
    # 1/S * 1/4
    conv3 = _maxpooling_conv(INITIAL_FILTER*4, 3, n_conv=2, activation='relu', last_activation=None)(conv2)
    # 1/S * 1/8
    conv4 = _maxpooling_conv(INITIAL_FILTER*8, 3, n_conv=2, activation='relu', last_activation=None)(conv3)
    # 1/S * 1/16
    conv5 = _maxpooling_conv(INITIAL_FILTER*16, 3, n_conv=2, activation='relu', last_activation=None)(conv4)

    # 1/S * 1/8
    upconv4 = _upsampling_conv_concate_conv(INITIAL_FILTER*8, 3, n_conv=2, activation='relu', last_activation=None)([conv5, conv4])
    # 1/S * 1/4
    upconv3 = _upsampling_conv_concate_conv(INITIAL_FILTER*4, 3, n_conv=2, activation='relu', last_activation=None)([upconv4, conv3])
    # 1/S * 1/2
    upconv2 = _upsampling_conv_concate_conv(INITIAL_FILTER*2, 3, n_conv=2, activation='relu', last_activation=None)([upconv3, conv2])
    # 1/S
    upconv1 = _upsampling_conv_concate_conv(INITIAL_FILTER, 3, n_conv=2, activation='relu', last_activation=None)([upconv2, conv1])

    # 1
    oup = UpSampling2D(size=(2,2))(upconv1)
    oup = Conv2D(num_class, 1, activation = 'sigmoid')(oup)

    model = Model(input = inputs, output = oup)

    return model

def unet_v2(input_shape, num_class):
    INITIAL_FILTER = 64
    INITIAL_KERNEL_SIZE = 7
    INITIAL_STRIDE = 2
    N_CONV = 2

    # 1
    inputs = Input(input_shape)
    x = inputs

    x = _conv_bn_relu(filters=INITIAL_FILTER, kernel_size=INITIAL_KERNEL_SIZE, strides=INITIAL_STRIDE, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 1/S
    conv1 = _maxpooling_conv(INITIAL_FILTER, 3, n_conv=N_CONV, activation='relu', last_activation=None, maxpooling=False)(x)
    # 1/S * 1/2
    conv2 = _maxpooling_conv(INITIAL_FILTER*2, 3, n_conv=N_CONV, activation='relu', last_activation=None)(conv1)
    # 1/S * 1/4
    conv3 = _maxpooling_conv(INITIAL_FILTER*4, 3, n_conv=N_CONV, activation='relu', last_activation=None)(conv2)
    # 1/S * 1/8
    conv4 = _maxpooling_conv(INITIAL_FILTER*8, 3, n_conv=N_CONV, activation='relu', last_activation=None)(conv3)

    # 1/S * 1/4
    upconv3 = _upsampling_conv_concate_conv(INITIAL_FILTER*4, 3, n_conv=N_CONV, activation='relu', last_activation=None)([conv4, conv3])
    # 1/S * 1/2
    upconv2 = _upsampling_conv_concate_conv(INITIAL_FILTER*2, 3, n_conv=N_CONV, activation='relu', last_activation=None)([upconv3, conv2])
    # 1/S
    upconv1 = _upsampling_conv_concate_conv(INITIAL_FILTER, 3, n_conv=N_CONV, activation='relu', last_activation=None)([upconv2, conv1])

    # 1
    #oup = UpSampling2D(size=(2,2))(upconv1)
    #oup = UpSampling2D(size=(2,2))(oup)
    #oup = Conv2D(num_class, 1, activation='sigmoid')(oup)

    oup = Conv2D(num_class, 1)(upconv1)
    now_shape = K.int_shape(oup)
    oup = Lambda(lambda x: tf.image.resize(x, (input_shape[0], input_shape[1])))(oup) #BILINEAR
    oup = Activation('sigmoid')(oup)

    model = Model(input = inputs, output = oup)

    return model

def unet_v3(input_shape, num_class):
    INITIAL_FILTER = 64
    INITIAL_KERNEL_SIZE = 7
    INITIAL_STRIDE = 2
    N_CONV = 2

    SHOURT_CUT='ave'
    DROP_ACT = None

    mp_conv_kwrgs1 = {'kernel_size':3, 'n_conv':N_CONV, 'activation':'relu', 'last_activation':None, 'drop_activation':DROP_ACT, 'maxpooling':False, 'shortcut':SHOURT_CUT}
    mp_conv_kwrgs2 = {'kernel_size':3, 'n_conv':N_CONV, 'activation':'relu', 'last_activation':None, 'drop_activation':DROP_ACT, 'maxpooling':True, 'shortcut':SHOURT_CUT}
    mp_conv_kwrgs3 = {'kernel_size':3, 'n_conv':N_CONV, 'activation':'relu', 'last_activation':None, 'drop_activation':DROP_ACT}


    # 1
    inputs = Input(input_shape)
    x = inputs

    x = _conv_bn_relu(filters=INITIAL_FILTER, kernel_size=INITIAL_KERNEL_SIZE, strides=INITIAL_STRIDE, activation='relu', drop_activation=DROP_ACT)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 1/S
    conv1 = _maxpooling_conv(INITIAL_FILTER, **mp_conv_kwrgs1)(x)
    # 1/S * 1/2
    conv2 = _maxpooling_conv(INITIAL_FILTER*2, **mp_conv_kwrgs2)(conv1)
    # 1/S * 1/4
    conv3 = _maxpooling_conv(INITIAL_FILTER*4, **mp_conv_kwrgs2)(conv2)
    # 1/S * 1/8
    conv4 = _maxpooling_conv(INITIAL_FILTER*8, **mp_conv_kwrgs2)(conv3)

    # 1/S * 1/4
    upconv3 = _upsampling_conv_concate_conv(INITIAL_FILTER*4, **mp_conv_kwrgs3)([conv4, conv3])
    # 1/S * 1/2
    upconv2 = _upsampling_conv_concate_conv(INITIAL_FILTER*2, **mp_conv_kwrgs3)([upconv3, conv2])
    # 1/S
    upconv1 = _upsampling_conv_concate_conv(INITIAL_FILTER, **mp_conv_kwrgs3)([upconv2, conv1])

    # 1
    #oup = UpSampling2D(size=(2,2))(upconv1)
    #oup = UpSampling2D(size=(2,2))(oup)
    #oup = Conv2D(num_class, 1, activation='sigmoid')(oup)

    oup = Conv2D(num_class, 1)(upconv1)
    now_shape = K.int_shape(oup)
    oup = Lambda(lambda _x: tf.image.resize(_x, (input_shape[0], input_shape[1])))(oup) #BILINEAR
    oup = Activation('sigmoid')(oup)

    model = Model(input = inputs, output = oup)

    return model