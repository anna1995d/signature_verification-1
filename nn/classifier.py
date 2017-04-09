from keras.layers import Dense
from keras.models import Sequential

from nn.logging import klogger


class Classifier(object):
    def __init__(self, activation, loss, optimizer, metrics):
        self.clsfr = Sequential()
        self.clsfr.add(Dense(1, input_dim=1, activation=activation))
        self.clsfr.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x, y, epochs, batch_size, verbose):
        return self.clsfr.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[klogger])

    def save(self, path):
        self.clsfr.save(path)
