import logging

from keras.callbacks import LambdaCallback
from keras.layers import RepeatVector
from keras.models import Sequential

logger = logging.getLogger(__name__)

klogger = LambdaCallback(
    on_epoch_end=lambda epoch, logs: logger.info('EPOCH #{epoch} end: loss: {loss}, accuracy: {accuracy}'.format(
        epoch=epoch, loss=logs['loss'], accuracy=logs['acc']
    )),
    on_batch_end=lambda batch, logs: logger.info('BATCH #{batch} end: loss: {loss}, accuracy: {accuracy}'.format(
        batch=batch, loss=logs['loss'], accuracy=logs['acc']
    ))
)


class Autoencoder(object):
    def __init__(self, cell, inp_max_len, inp_dim, enc_len, loss='mean_squared_error', optimizer='adam'):
        self.seq_autoenc = Sequential()
        self.seq_autoenc.add(cell(enc_len, input_shape=(inp_max_len, inp_dim), name='encoder'))
        self.seq_autoenc.add(RepeatVector(inp_max_len, name='repeater'))
        self.seq_autoenc.add(cell(inp_dim, return_sequences=True, name='decoder'))
        self.seq_autoenc.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def fit(self, tr_inp, epochs=10, batch_size=32, verbose=1):
        self.seq_autoenc.fit(tr_inp, tr_inp, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[klogger])

    def save(self, path):
        self.seq_autoenc.save(path)


class Encoder(object):
    def __init__(self, cell, inp_max_len, inp_dim, enc_len, loss='mean_squared_error', optimizer='adam'):
        self.encoder = Sequential()
        self.encoder.add(cell(enc_len, input_shape=(inp_max_len, inp_dim), name='encoder'))
        self.encoder.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def predict(self, inp, batch_size=32):
        return self.encoder.predict(inp, batch_size=batch_size)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)
