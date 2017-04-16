from keras.layers import RepeatVector, Masking
from keras.models import Sequential

from nn.logging import elogger, blogger


class Autoencoder(object):
    def __init__(self, cell, inp_max_len, inp_dim, enc_len, loss, optimizer, metrics, implementation, mask_value):
        self.seq_autoenc = Sequential()
        self.seq_autoenc.add(Masking(mask_value=mask_value, input_shape=(inp_max_len, inp_dim)))
        self.seq_autoenc.add(cell(enc_len, implementation=implementation, name='encoder'))
        self.seq_autoenc.add(RepeatVector(inp_max_len, name='repeater'))
        self.seq_autoenc.add(cell(inp_dim, return_sequences=True, implementation=implementation, name='decoder'))
        self.seq_autoenc.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x, y, epochs, batch_size, verbose):
        self.seq_autoenc.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[elogger, blogger])

    def save(self, path):
        self.seq_autoenc.save(path)


class Encoder(object):
    def __init__(self, cell, inp_max_len, inp_dim, enc_len, loss, optimizer, metrics, implementation, mask_value):
        self.encoder = Sequential()
        self.encoder.add(Masking(mask_value=mask_value, input_shape=(inp_max_len, inp_dim)))
        self.encoder.add(cell(enc_len, implementation=implementation, name='encoder'))
        self.encoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def predict(self, inp, batch_size):
        return self.encoder.predict(inp, batch_size=batch_size)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)
