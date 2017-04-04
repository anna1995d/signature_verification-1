from keras.layers import RepeatVector
from keras.models import Sequential

from rnn.logging import klogger


class Autoencoder(object):
    def __init__(self, cell, inp_max_len, inp_dim, enc_len, loss, optimizer, metrics, implementation):
        self.seq_autoenc = Sequential()
        self.seq_autoenc.add(
            cell(enc_len, input_shape=(inp_max_len, inp_dim), implementation=implementation, name='encoder')
        )
        self.seq_autoenc.add(RepeatVector(inp_max_len, name='repeater'))
        self.seq_autoenc.add(cell(inp_dim, return_sequences=True, implementation=implementation, name='decoder'))
        self.seq_autoenc.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x, y, epochs=10, batch_size=32, verbose=1):
        self.seq_autoenc.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[klogger])

    def save(self, path):
        self.seq_autoenc.save(path)


class Encoder(object):
    def __init__(self, cell, inp_max_len, inp_dim, enc_len, loss, optimizer, metrics, implementation):
        self.encoder = Sequential()
        self.encoder.add(
            cell(enc_len, input_shape=(inp_max_len, inp_dim), implementation=implementation, name='encoder')
        )
        self.encoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def predict(self, inp, batch_size=32):
        return self.encoder.predict(inp, batch_size=batch_size)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)
