from keras.layers import LSTM, RepeatVector, Dense, Dropout
from keras.models import Sequential


class LSTMAutoencoder(object):

    def __init__(self, input_length, input_dimension, encoded_length):
        self.sequence_autoencoder = Sequential()
        self.sequence_autoencoder.add(
            LSTM(output_dim=encoded_length, input_shape=(input_length, input_dimension), name='encoder')
        )
        self.sequence_autoencoder.add(RepeatVector(input_length, name='repeater'))
        self.sequence_autoencoder.add(LSTM(input_dimension, return_sequences=True, name='decoder'))
        self.sequence_autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def fit(self, train_input, nb_epoch=10, batch_size=32, verbose=1):
        self.sequence_autoencoder.fit(
            train_input, train_input, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose
        )

    def save(self, save_path):
        self.sequence_autoencoder.save(save_path)


class LSTMEncoder(object):

    def __init__(self, input_length, input_dimension, encoded_length):
        self.encoder = Sequential()
        self.encoder.add(LSTM(encoded_length, input_shape=(input_length, input_dimension), name='encoder'))
        self.encoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def predict(self, inp, batch_size=32):
        return self.encoder.predict(inp, batch_size=batch_size)

    def load(self, load_path):
        self.encoder.load_weights(load_path, by_name=True)


class Classifier(object):

    def __init__(self, encoded_length, dropout=0.5):
        self.classifier = Sequential()
        self.classifier.add(
            Dense(encoded_length, input_shape=(encoded_length,), init='uniform', activation='sigmoid')
        )
        self.classifier.add(Dropout(dropout))
        self.classifier.add(
            Dense(encoded_length, init='uniform', activation='sigmoid')
        )
        self.classifier.add(Dropout(dropout))
        self.classifier.add(
            Dense(1, init='uniform', activation='sigmoid')
        )
        self.classifier.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])

    def fit(self, train_x, train_y, nb_epoch=10, batch_size=32, verbose=1):
        self.classifier.fit(train_x, train_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)

    def save(self, save_path):
        self.classifier.save(save_path)

    def load(self, load_path):
        self.classifier.load_weights(load_path, by_name=True)

    def predict(self, inp, batch_size=32):
        return self.classifier.predict(inp, batch_size=batch_size)

    def evaluate(self, test_x, train_y, batch_size=32):
        return self.classifier.evaluate(test_x, train_y, batch_size=batch_size)
