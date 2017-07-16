import keras.backend as K
from keras import initializers
from keras.layers import Layer


class Attention(Layer):
    def __init__(self, **kwargs):
        self.kernel = None
        self.supports_masking = True
        self.kernel_initializer = initializers.get('glorot_uniform')
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.kernel = self.add_weight('kernel', (input_shape[-1],) * 2, initializer=self.kernel_initializer)
        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        eij = K.tanh(K.dot(inputs, self.kernel))
        ai = K.exp(eij)
        ai /= K.cast(K.sum(ai, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return K.sum(inputs * ai, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
