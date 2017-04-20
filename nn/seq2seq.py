from keras.layers import Masking
from keras.models import Sequential

from nn.logging import elogger, blogger


class Autoencoder(object):
    def __init__(self, cell, inp_dim, enc_len, ctxl_len, loss, optimizer, metrics, implementation, mask_value):
        self.seq_autoenc = Sequential()
        self.seq_autoenc.add(Masking(mask_value=mask_value, input_shape=(None, inp_dim)))
        self.seq_autoenc.add(cell(enc_len, return_sequences=True, implementation=implementation, name='encoder_1'))
        self.seq_autoenc.add(cell(ctxl_len, return_sequences=True, implementation=implementation, name='encoder_2'))
        self.seq_autoenc.add(cell(ctxl_len, return_sequences=True, implementation=implementation, name='decoder_1'))
        self.seq_autoenc.add(cell(inp_dim, return_sequences=True, implementation=implementation, name='decoder_2'))
        self.seq_autoenc.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x, y, epochs, batch_size, verbose):
        self.seq_autoenc.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[elogger, blogger])

    def save(self, path):
        self.seq_autoenc.save(path)


class Encoder(object):
    def __init__(self, cell, inp_dim, enc_len, ctxl_len, loss, optimizer, metrics, implementation, mask_value):
        self.encoder = Sequential()
        self.encoder.add(Masking(mask_value=mask_value, input_shape=(None, inp_dim)))
        self.encoder.add(cell(enc_len, return_sequences=True, implementation=implementation, name='encoder_1'))
        self.encoder.add(cell(ctxl_len, implementation=implementation, name='encoder_2'))
        self.encoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def predict(self, inp):
        return self.encoder.predict(inp)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)
