from keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from keras.models import Model, Sequential


class LSTMAutoencoder(object):

    def __init__(self, input_length, input_dimension, encoded_length):
        inputs = Input(shape=(input_length, input_dimension))
        encode_rnn = LSTM(encoded_length, name='encoder')(inputs)
        repeat_encoding_rnn = RepeatVector(input_length, name='repeater')(encode_rnn)
        decode_rnn = LSTM(input_dimension, return_sequences=True, name='decoder')(repeat_encoding_rnn)

        self.sequence_autoencoder = Model(inputs, decode_rnn)
        self.sequence_autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def fit(self, train_input, output_path, nb_epoch=10, batch_size=32):
        self.sequence_autoencoder.fit(train_input, train_input, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
        self.sequence_autoencoder.save(output_path)


class LSTMEncoder(object):

    def __init__(self, input_length, input_dimension, encoded_length, load_path):
        inputs = Input(shape=(input_length, input_dimension))
        encode_rnn = LSTM(encoded_length, name='encoder')(inputs)

        self.encoder = Model(inputs, encode_rnn)
        self.encoder.load_weights(load_path, by_name='encoder')

    def predict(self, inp, batch_size=32):
        return self.encoder.predict(inp, batch_size=batch_size)


class Classifier(object):

    def __init__(self, input_dimension, encoded_length, dropout):
        self.classifier = Sequential()
        self.classifier.add(Dense(encoded_length, input_dim=input_dimension, init='uniform', activation='relu'))
        self.classifier.add(Dropout(dropout))
        self.classifier.add(Dense(encoded_length, activation='relu'))
        self.classifier.add(Dropout(dropout))
        self.classifier.add(Dense(1, activation='sigmoid'))
        self.classifier.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def fit(self, train_x, train_y, output_path, nb_epoch=10, batch_size=32):
        self.classifier.fit(train_x, train_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
        self.classifier.save(output_path)

    def predict(self, inp, batch_size=32):
        return self.classifier.predict(inp, batch_size=batch_size)

    def evaluate(self, test_x, train_y, batch_size=32):
        return self.classifier.evaluate(test_x, train_y, batch_size=batch_size)
