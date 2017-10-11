import numpy as np
from keras.utils import Sequence

from data import Data


class CustomSequence(Sequence):
    def __init__(self, length, max_length, path):
        self.length = length
        self.max_length = max_length
        self.path = path
        self.batch = 0

    def __len__(self):
        return self.length

    def __getitem__(self, batch):
        self.batch += 1
        f = np.load(self.path.format(batch))
        return f['x'].astype(np.float32), f['y'].astype(np.float32)

    def __next__(self):
        return self.__getitem__(self.batch)

    def on_epoch_end(self):
        pass


class CustomTwoBranchSequence(Sequence):
    def __init__(self, length, max_length, path):
        self.length = length
        self.max_length = max_length
        self.path = path
        self.batch = 0

    def __len__(self):
        return self.length

    def __getitem__(self, batch):
        f = np.load(self.path.format(batch)).astype(np.float32)
        return [f['x_0'].astype(np.float32), f['x_1'].astype(np.float32)], f['y'].astype(np.float32)

    def __next__(self):
        return self.__getitem__(self.batch)

    def on_epoch_end(self):
        pass


def get_generator(x, y, path, batch_size):
    if type(x) == list or type(y) == list:
        batches = (len(x[0]) + (batch_size - 1)) // batch_size
        for batch in range(batches):
            start = batch * batch_size
            end = batch * (batch_size + 1) if batch * (batch_size + 1) >= len(x[0]) else None
            np.savez(path.format(batch), x_0=x[0][start:end], x_1=x[1][start:end], y=x[start:end])
        return CustomTwoBranchSequence(batches, x.shape[1], path)
    else:
        batches = (len(x) + (batch_size - 1)) // batch_size
        for batch in range(batches):
            start = batch * batch_size
            end = batch * (batch_size + 1) if batch * (batch_size + 1) >= len(x) else None
            np.savez(path.format(batch), x=x[start:end], y=x[start:end])
        return CustomSequence(batches, x.shape[1], path)


DATA = Data()
