from keras.layers import Masking, InputLayer, RepeatVector
from keras.models import Sequential
from sklearn import svm
from sklearn.externals import joblib

from seq2seq.layers import AttentionWithContext
from seq2seq.logging import elogger, blogger, tblogger


# TODO: Add EarlyStopping Callback
# TODO: Add LearningRateScheduler if it is useful
class Autoencoder(object):
    def __init__(self, cell, inp_dim, max_len, earc, darc, loss, optimizer, metrics, implementation, mask_value):
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

        # Attention
        self.seq_autoenc.add(AttentionWithContext())

        # Decoder
        self.seq_autoenc.add(RepeatVector(max_len))
        for i, ln in enumerate(darc):
            self.seq_autoenc.add(cell(
                ln,
                return_sequences=True,
                implementation=implementation,
                name='decoder_{index}'.format(index=i)
            ))

        self.seq_autoenc.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x, y, epochs, batch_size, verbose):
        self.seq_autoenc.fit(
            x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[elogger, blogger, tblogger]
        )

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
                return_sequences=i != len(earc) - 1,
                implementation=implementation,
                name='encoder_{index}'.format(index=i)
            ))

        self.encoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def predict(self, inp):
        return self.encoder.predict(inp)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)


class LinearSVC(svm.LinearSVC):
    def save(self, path):
        joblib.dump(self, path)
