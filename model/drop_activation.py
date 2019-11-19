"""
Reference:
    drop-activation original paper : https://arxiv.org/abs/1811.05850
    implementation in keras by JGuillaumin : https://github.com/JGuillaumin/drop_activation_tf
"""

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class DropActivation(Layer):
    def __init__(self, rate=0.05, seed=None, activation_func=None, **kwargs):
        """
        Args:
            rate : drop rate,
            seed : random seed,
            activation_func : function of activation function. use K.relu if is None.
        """
        super(DropActivation, self).__init__(**kwargs)
        self.supports_masking = True

        self.RATE = rate
        self.SEED = seed
        self.ACTIVATION_FUNC = activation_func if activation_func is not None else K.relu

        return

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        
        def oup_tensor_in_training():
            # input tensor
            inp_tensor = tf.convert_to_tensor(inputs)
            # 0:drop, 1:retain
            mask = K.ones_like(inp_tensor, dtype=K.floatx())
            mask = K.dropout(mask, level=self.RATE, seed=self.SEED)
            # output tensor
            masked = (1.0 - mask) * inp_tensor
            not_masked = mask * self.ACTIVATION_FUNC(inp_tensor)
            oup_tensor = masked + not_masked
            return oup_tensor

        def oup_tensor_in_test():
            # input tensor
            inp_tensor = tf.convert_to_tensor(inputs)
            # mask = expectation value of retain rate
            mask = 1.0 - self.RATE
            # output tensor
            masked = (1.0 - mask) * inp_tensor
            not_masked = mask * self.ACTIVATION_FUNC(inp_tensor)
            oup_tensor = masked + not_masked
            return oup_tensor

        outputs = K.in_train_phase(oup_tensor_in_training, oup_tensor_in_test, training)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.RATE,
            'seed': self.SEED,
            'activation_func': self.ACTIVATION_FUNC,
        }

        base_config = super(DropActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
