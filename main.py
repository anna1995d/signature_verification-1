#!/usr/bin/env python

import os
import itertools

import numpy as np

from rnn.seq2seq_autoencoder import LSTMAutoencoder, LSTMEncoder

from dtw.data import Data
from dtw.dtw import DTW

from keras.preprocessing import sequence

PATH = os.path.dirname(__file__)


def generate_data():
    return Data(
        user_count=user_count,
        genuine_sample_count=genuine_sample_count,
        forged_sample_count=forged_sample_count,
        forger_count=forger_count,
        genuine_file_path_template=genuine_file_path_template,
        forged_file_path_template=forged_file_path_template,
        frame_size=frame_size
    )


def run_dtw():
    with open(PATH + 'genuine.txt', 'a') as f:
        for user in range(user_count):
            for x, y in data.get_combinations(user, forged=False):
                dtw = DTW(x, y, window_size, DTW.euclidean)
                f.write(str(dtw.calculate()) + '\n')

    with open(PATH + 'forged.txt', 'a') as f:
        for user in range(user_count):
            for x, y in data.get_combinations(user, forged=True):
                dtw = DTW(x, y, window_size, DTW.euclidean)
                f.write(str(dtw.calculate()) + '\n')


def generate_masked_input_data():
    max_lengths = sorted([
        _.shape[0] for _ in itertools.chain.from_iterable(data.genuine_extracted_data + data.forged_extracted_data)
    ])

    res = np.array([
        _ for _ in itertools.chain.from_iterable(data.genuine_extracted_data + data.forged_extracted_data)
        if _.shape[0] < max_lengths[-1]
    ])
    return sequence.pad_sequences(res, maxlen=max_lengths[-2]), max_lengths[-2]


def generate_padded_input_data():
    return np.concatenate([
        np.concatenate(data.framed_genuine_extracted_data), np.concatenate(data.framed_forged_extracted_data)
    ]), frame_size


def run_autoencoder():
    autoencoder = LSTMAutoencoder(input_length=max_length, input_dimension=2, encoded_length=50)
    autoencoder.fit(train_input=train_input, output_path=rnn_output_path, nb_epoch=nb_epoch)
    return autoencoder


def get_encoder():
    run_autoencoder()
    return LSTMEncoder(input_length=max_length, input_dimension=2, encoded_length=50, load_path=rnn_output_path)


if __name__ == '__main__':
    genuine_file_path_template = os.path.join(PATH, 'data/Genuine/{user}/{sample}_{user}.HWR')
    forged_file_path_template = os.path.join(PATH, 'data/Forged/{user}/{sample}_{forger}_{user}.HWR')

    rnn_output_path = os.path.join(PATH, 'model.dat')

    window_size = 4
    user_count = 11
    genuine_sample_count = 42
    forged_sample_count = 36
    forger_count = 4
    frame_size = 100
    nb_epoch = 5

    data = generate_data()
    train_input, max_length = generate_masked_input_data()

    encoder = get_encoder()
