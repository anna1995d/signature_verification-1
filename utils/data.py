from keras.utils import Sequence

from data import Data


class CustomSequence(Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.x) + (self.batch_size - 1)) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = idx * (self.batch_size + 1) if idx * (self.batch_size + 1) >= len(self.x) else None
        return self.x[start:end], self.y[start:end]

    def on_epoch_end(self):
        pass


class CustomTwoBranchSequence(Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.x[0]) + (self.batch_size - 1)) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = idx * (self.batch_size + 1) if idx * (self.batch_size + 1) >= len(self.x) else None
        return [self.x[0][start:end], self.x[1][start:end]], self.y[start:end]

    def on_epoch_end(self):
        pass


DATA = Data()
