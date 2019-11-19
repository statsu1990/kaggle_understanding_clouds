from keras import backend as K
from keras.layers import Layer
from keras import regularizers

import tensorflow as tf

def regularizers_l2norm_to_const_wrapper(c, coef, axis):
    def regularizers_l2norm_to_const(weight_matrix):
        l2norm = tf.norm(weight_matrix, ord=2, axis=axis)
        diff = tf.square(l2norm/c - 1)
        sum_diff = tf.reduce_sum(diff)
        return coef * sum_diff
    return regularizers_l2norm_to_const

class CosSimilarityWithFeatvec(Layer):
    def __init__(self, n_vec, regu_coef=None, **kwargs):
        super(CosSimilarityWithFeatvec, self).__init__(**kwargs)
        self.n_vec = n_vec

        if regu_coef is None:
            self.regularizer = None
        else:
            self.regularizer = regularizers_l2norm_to_const_wrapper(c=1, coef=regu_coef, axis=0)

    def build(self, input_shape):
        """
        Args:
            input_shape = (Batch, H, W, M)
        """
        super(CosSimilarityWithFeatvec, self).build(input_shape)
        # feature vectors shape = (M, N_vec)
        self.featvecs = self.add_weight(name='feature_vecs',
                                shape=(input_shape[-1], self.n_vec),
                                initializer='lecun_normal',
                                trainable=True,
                                regularizer=self.regularizer
                                )

    def call(self, inputs):
        """
        Args:
            input_shape = (Batch, H, W, M)
        """
        # l2 normalized input
        x = inputs
        x = tf.math.l2_normalize(x, axis=-1)

        # l2 normalized feature vecs
        fvs = tf.math.l2_normalize(self.featvecs, axis=0)

        # (Batch, H, W, M) * (M, N_vec) = (Batch, H, W, N_vec)
        cos_simi = K.dot(x, fvs)
        
        return cos_simi

    def compute_output_shape(self, input_shape):
        """
        Returns:
            (Batch, H, W, N_vec)
        """
        return input_shape[:-1] + (self.n_vec,)

class ActivityRegularization_L2NormToConst(Layer):
    """Layer that applies an update to the cost function based input activity.
    # Arguments
        
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, c, coef, axis, **kwargs):
        super(ActivityRegularization_L2NormToConst, self).__init__(**kwargs)
        self.supports_masking = True
        self.c = c
        self.coef = coef
        self.axis = axis
        self.activity_regularizer = regularizers_l2norm_to_const_wrapper(c=c, coef=coef, axis=axis)

    def get_config(self):
        config = {'c': self.c,
                  'coef': self.coef,
                  'axis': self.axis,
                  }
        base_config = super(ActivityRegularization_L2NormToConst, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape