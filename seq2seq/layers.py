import keras.backend as K
from keras import initializers, regularizers, constraints
from keras.layers import Layer


class AttentionWithContext(Layer):
    def __init__(self,
                 kernel_regularizer=None, align_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, align_constraint=None, bias_constraint=None,
                 use_bias=True, **kwargs):

        self.kernel = None
        self.bias = None
        self.align = None

        self.supports_masking = True
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.align_regularizer = regularizers.get(align_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.align_constraint = constraints.get(align_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], input_shape[-1],),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(input_shape[-1],),
                initializer='zero',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        self.align = self.add_weight(
            name='align',
            shape=(input_shape[-1],),
            initializer=self.kernel_initializer,
            regularizer=self.align_regularizer,
            constraint=self.align_constraint
        )

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        uit = K.tanh(K.dot(inputs, self.kernel) + (self.bias if self.use_bias else 0))
        ait = K.sum(uit * self.align, axis=2) if K.backend() == 'tensorflow' else K.dot(uit, self.align)
        a = K.exp(ait) * (K.cast(mask, K.floatx()) if mask is not None else 1)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        return K.sum(inputs * a, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class Attention(Layer):
    def __init__(self,
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 use_bias=True, **kwargs):
        self.kernel = None
        self.bias = None

        self.supports_masking = True
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], input_shape[-1],),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(input_shape[-1],),
                initializer='zero',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        super(Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        eij = K.tanh(K.dot(inputs, self.kernel) + (self.bias if self.use_bias else 0))
        ai = K.exp(eij) * K.expand_dims(K.cast(mask, K.floatx()) if mask is not None else 1)
        a = ai / K.cast(K.sum(ai, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return K.sum(inputs * a, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
