import logging

from keras.callbacks import LambdaCallback
from keras.layers import RepeatVector
from keras.models import Sequential

logger = logging.getLogger(__name__)

keras_logger = LambdaCallback(
    on_epoch_end=lambda epoch, logs: logger.info('EPOCH #{epoch} end: loss: {loss}, accuracy: {accuracy}'.format(
        epoch=epoch, loss=logs['loss'], accuracy=logs['acc']
    )),
    on_batch_end=lambda batch, logs: logger.info('BATCH #{batch} end: loss: {loss}, accuracy: {accuracy}'.format(
        batch=batch, loss=logs['loss'], accuracy=logs['acc']
    ))
)


class AutoEncoder(object):
    def __init__(self, cell, inp_max_len, inp_dim, enc_len, loss='mean_squared_error', optimizer='adam'):
        self.seq_auto_enc = Sequential()
        self.seq_auto_enc.add(cell(output_dim=enc_len, input_shape=(inp_max_len, inp_dim), name='encoder'))
        self.seq_auto_enc.add(RepeatVector(inp_max_len, name='repeater'))
        self.seq_auto_enc.add(cell(inp_dim, return_sequences=True, name='decoder'))
        self.seq_auto_enc.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def fit(self, tr_inp, nb_epoch=10, batch_size=32, verbose=1):
        self.seq_auto_enc.fit(tr_inp, tr_inp, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose,
                              callbacks=[keras_logger])

    def save(self, path):
        self.seq_auto_enc.save(path)


class Encoder(object):
    def __init__(self, cell, inp_max_len, inp_dim, enc_len, loss='mean_squared_error', optimizer='adam'):
        self.encoder = Sequential()
        self.encoder.add(cell(enc_len, input_shape=(inp_max_len, inp_dim), name='encoder'))
        self.encoder.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def predict(self, inp, batch_size=32):
        return self.encoder.predict(inp, batch_size=batch_size)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)
