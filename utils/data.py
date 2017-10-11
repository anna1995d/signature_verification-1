import numpy as np
from keras.utils import Sequence

from data import Data


class CustomSequence(Sequence):
    def __init__(self, length, max_length, path):
        self.length = length
        self.max_length = max_length
        self.path = path

    def __len__(self):
        return self.length

    def __getitem__(self, batch):
        f = np.load(self.path.format(batch))
        return f['x'], f['y']

    def on_epoch_end(self):
        pass


class CustomTwoBranchSequence(Sequence):
    def __init__(self, length, max_length, path):
        self.length = length
        self.max_length = max_length
        self.path = path

    def __len__(self):
        return self.length

    def __getitem__(self, batch):
        f = np.load(self.path.format(batch))
        return [f['x_0'], f['x_1']], f['y']

    def on_epoch_end(self):
        pass


def get_generator(x, y, path, batch_size):
    if type(x) == list or type(y) == list:
        batches = (len(x[0]) + (batch_size - 1)) // batch_size
        for batch in range(batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size if (batch + 1) * batch_size < len(x[0]) else None
            np.savez(path.format(batch), x_0=x[0][start:end], x_1=x[1][start:end], y=y[start:end])
        return CustomTwoBranchSequence(batches, x[0].shape[1], path)
    else:
        batches = (len(x) + (batch_size - 1)) // batch_size
        for batch in range(batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size if (batch + 1) * batch_size < len(x) else None
            np.savez(path.format(batch), x=x[start:end], y=y[start:end])
        return CustomSequence(batches, x.shape[1], path)


DATA = Data()
