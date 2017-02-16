#!/usr/bin/env python

import os
import itertools
import numpy as np

PATH = os.path.dirname(__file__)


if __name__ == '__main__':
    with open(PATH + 'genuine.txt', 'r') as f:
        px = np.sort(map(int, f.read().split('\n')))

    with open(PATH + 'forged.txt', 'r') as f:
        nx = np.sort(map(int, f.read().split('\n')))

    threshold = 0
    min_dist = np.matmul(px, px.T) + np.matmul(nx, nx.T)
    for i in itertools.chain(px, nx):
        filtered_px = px[px < i]
        filtered_nx = nx[nx < i]
        dist = np.linalg.norm(filtered_px - np.full(filtered_px.shape, i)) + \
            np.linalg.norm(filtered_nx - np.full(filtered_nx.shape, i))

        if dist < min_dist:
            min_dist = dist
            threshold = i

    with open(PATH + 'threshold.txt', 'a') as f:
        f.write(str(threshold) + '\n')
        f.write(str(min_dist) + '\n')
