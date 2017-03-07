from keras.layers import LSTM, RepeatVector
from keras.models import Sequential


class LSTMAutoEncoder(object):
    def __init__(self, inp_max_len, inp_dim, enc_len):
        self.seq_auto_enc = Sequential()
        self.seq_auto_enc.add(LSTM(output_dim=enc_len, input_shape=(inp_max_len, inp_dim), name='encoder'))
        self.seq_auto_enc.add(RepeatVector(inp_max_len, name='repeater'))
        self.seq_auto_enc.add(LSTM(inp_dim, return_sequences=True, name='decoder'))
        self.seq_auto_enc.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def fit(self, tr_inp, nb_epoch=10, batch_size=32, verbose=1):
        self.seq_auto_enc.fit(tr_inp, tr_inp, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)

    def save(self, path):
        self.seq_auto_enc.save(path)


class LSTMEncoder(object):
    def __init__(self, inp_max_len, inp_dim, enc_len):
        self.encoder = Sequential()
        self.encoder.add(LSTM(enc_len, input_shape=(inp_max_len, inp_dim), name='encoder'))
        self.encoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def predict(self, inp, batch_size=32):
        return self.encoder.predict(inp, batch_size=batch_size)

    def load(self, path):
        self.encoder.load_weights(path, by_name=True)
