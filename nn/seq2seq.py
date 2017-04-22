from keras.layers import Masking, InputLayer
from keras.models import Sequential

from nn.logging import elogger, blogger


class Autoencoder(object):
    def __init__(self, cell, inp_dim, earc, darc, loss, optimizer, metrics, implementation, mask_value):
        self.seq_autoenc = Sequential()

        # Input
        self.seq_autoenc.add(InputLayer(input_shape=(None, inp_dim), name='input'))
        self.seq_autoenc.add(Masking(mask_value=mask_value, name='mask'))

        # Encoder
        for i, ln in enumerate(earc):
            self.seq_autoenc.add(cell(
                ln,
                return_sequences=True,
                implementation=implementation,
                name='encoder_{index}'.format(index=i)
            ))

        # Decoder
        for i, ln in enumerate(darc):
            self.seq_autoenc.add(cell(
                ln,
                return_sequences=True,
                implementation=implementation,
                name='decoder_{index}'.format(index=i)
            ))

        self.seq_autoenc.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x, y, epochs, batch_size, verbose):
        self.seq_autoenc.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[elogger, blogger])

    def save(self, path):
        self.seq_autoenc.save(path)


class Encoder(object):
    def __init__(self, cell, inp_dim, earc, loss, optimizer, metrics, implementation, mask_value):
        self.encoder = Sequential()

        # Input
        self.encoder.add(InputLayer(input_shape=(None, inp_dim), name='input'))
        self.encoder.add(Masking(mask_value=mask_value, name='mask'))

        # Encoder
        for i, ln in enumerate(earc):
            self.encoder.add(cell(
                ln,
                return_sequences=True if i != len(earc) - 1 else False,
                implementation=implementation,
                name='encoder_{index}'.format(index=i)
            ))

        self.encoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def predict(self, inp):
        return self.encoder.predict(inp)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)
