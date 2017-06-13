import itertools

import numpy as np


def compute_distances(x, y=None):
    if y is None:
        dists = list(map(lambda vecs: np.linalg.norm(vecs[0] - vecs[1]), itertools.product(x, x)))
        return np.delete(dists, range(0, len(x) ** 2, len(x) + 1)).reshape((len(x), -1))
    else:
        dists = list(map(lambda vecs: np.linalg.norm(vecs[0] - vecs[1]), itertools.product(x, y)))
        return np.reshape(dists, (len(x), -1))
