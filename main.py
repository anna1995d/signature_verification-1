#!/usr/bin/env python

import os
import itertools

import numpy as np

from rnn.seq2seq_autoencoder import LSTMAutoencoder, LSTMEncoder, Classifier

from dtw.data import Data
from dtw.dtw import DTW

from keras.preprocessing import sequence

PATH = os.path.dirname(__file__)

genuine_file_path_template = os.path.join(PATH, 'data/Genuine/{user}/{sample}_{user}.HWR')
forged_file_path_template = os.path.join(PATH, 'data/Forged/{user}/{sample}_{forger}_{user}.HWR')

save_path_template = os.path.join(PATH, '{name}_model.dat')

window_size = 4
user_count = 11
genuine_sample_count = 42
forged_sample_count = 36
forger_count = 4
frame_size = 100

encoded_length = 100
input_dim = 2
ae_nb_epoch = 100
c_nb_epoch = 100


def generate_data():
    d = Data(
        user_count=user_count,
        genuine_sample_count=genuine_sample_count,
        forged_sample_count=forged_sample_count,
        forger_count=forger_count,
        genuine_file_path_template=genuine_file_path_template,
        forged_file_path_template=forged_file_path_template,
        frame_size=frame_size
    )
    return d


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


def get_autoencoder():
    ae = LSTMAutoencoder(input_length=max_length, input_dimension=input_dim, encoded_length=encoded_length)
    ae.fit(train_input=train_input_x, nb_epoch=ae_nb_epoch)
    ae.save(save_path=save_path_template.format(name='autoencoder'))
    return ae


def get_encoder():
    e = LSTMEncoder(input_length=max_length, input_dimension=input_dim, encoded_length=encoded_length)
    e.load(load_path=save_path_template.format(name='autoencoder'))
    return e


def get_classifier():
    c = Classifier(encoded_length=encoded_length)
    c.fit(train_x=encoded_input_x, train_y=train_input_y, nb_epoch=c_nb_epoch)
    c.save(save_path=save_path_template.format(name='classifier'))
    return c


def generate_padded_input_data():
    max_lengths = sorted([
        _.shape[0] for _ in itertools.chain.from_iterable(data.genuine_extracted_data + data.forged_extracted_data)
    ])

    res_x = [
        _ for _ in itertools.chain.from_iterable(data.genuine_extracted_data)
        if _.shape[0] < max_lengths[-1]
    ]
    res_y = np.ones((len(res_x), 1))

    res_x += [
        _ for _ in itertools.chain.from_iterable(data.forged_extracted_data)
        if _.shape[0] < max_lengths[-1]
    ]
    res_y = np.concatenate((res_y, np.zeros((len(res_x) - len(res_y), 1))))

    return sequence.pad_sequences(np.array(res_x), maxlen=max_lengths[-2]), np.array(res_y), max_lengths[-2]


def generate_sliced_input_data():
    return np.concatenate([
        np.concatenate(data.framed_genuine_extracted_data), np.concatenate(data.framed_forged_extracted_data)
    ]), frame_size


if __name__ == '__main__':
    data = generate_data()

    train_input_x, train_input_y, max_length = generate_padded_input_data()

    autoencoder = get_autoencoder()
    encoder = get_encoder()
    encoded_input_x = encoder.predict(train_input_x)

    classifier = get_classifier()
