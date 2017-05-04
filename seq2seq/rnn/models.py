from keras.layers import Masking, InputLayer, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from sklearn import svm
from sklearn.externals import joblib

from seq2seq.logging import elogger
from seq2seq.rnn.layers import AttentionWithContext
from seq2seq.rnn.logging import rnn_tblogger


# TODO: Add EarlyStopping Callback
# TODO: Add LearningRateScheduler if it is useful
class Autoencoder(object):
    def __init__(self, cell, bidir, bidir_mrgm, inp_dim, max_len, earc, darc, msk_val, ccfg, lcfg):
        self.seq_autoenc = Sequential()

        # Input
        self.seq_autoenc.add(InputLayer(input_shape=(None, inp_dim), name='input'))
        self.seq_autoenc.add(Masking(mask_value=msk_val, name='mask'))

        # Encoder
        for i, ln in enumerate(earc):
            c = cell(ln, **lcfg, name='encoder_{index}'.format(index=i))
            self.seq_autoenc.add(Bidirectional(c, merge_mode=bidir_mrgm) if bidir else c)

        # Attention
        self.seq_autoenc.add(AttentionWithContext())

        # Decoder
        self.seq_autoenc.add(RepeatVector(max_len))
        for i, ln in enumerate(darc):
            c = cell(ln, **lcfg, name='decoder_{index}'.format(index=i))
            self.seq_autoenc.add(Bidirectional(c, merge_mode=bidir_mrgm) if bidir else c)

        self.seq_autoenc.compile(**ccfg)

    def fit(self, x, y, epochs, batch_size, verbose, usr_num):
        self.seq_autoenc.fit(
            x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[elogger, rnn_tblogger(usr_num)]
        )

    def save(self, path):
        self.seq_autoenc.save(path)


class Encoder(object):
    def __init__(self, cell, bidir, bidir_mrgm, inp_dim, earc, msk_val, ccfg, lcfg):
        self.encoder = Sequential()

        # Input
        self.encoder.add(InputLayer(input_shape=(None, inp_dim), name='input'))
        self.encoder.add(Masking(mask_value=msk_val, name='mask'))

        # Encoder
        for i, ln in enumerate(earc):
            c = cell(ln, **lcfg, name='encoder_{index}'.format(index=i))
            c.return_sequences = (i != len(earc) - 1)
            self.encoder.add(Bidirectional(c, merge_mode=bidir_mrgm) if bidir else c)

        self.encoder.compile(**ccfg)

    def predict(self, inp):
        return self.encoder.predict(inp)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)


class LinearSVC(svm.LinearSVC):
    def save(self, path):
        joblib.dump(self, path)
