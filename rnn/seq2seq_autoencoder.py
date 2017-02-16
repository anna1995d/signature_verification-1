from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model


class LSTMAutoencoder(object):

    def __init__(self, input_length, input_dimension, encoded_length):
        inputs = Input(shape=(input_length, input_dimension))
        encode_rnn = LSTM(encoded_length)(inputs)
        repeat_encoding_rnn = RepeatVector(input_length)(encode_rnn)
        decode_rnn = LSTM(input_dimension, return_sequences=True)(repeat_encoding_rnn)

        self.sequence_autoencoder = Model(inputs, decode_rnn)
        self.sequence_autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def fit(self, train_input, output_path, nb_epoch=10, batch_size=32):
        self.sequence_autoencoder.fit(train_input, train_input, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
        self.sequence_autoencoder.save(output_path)
