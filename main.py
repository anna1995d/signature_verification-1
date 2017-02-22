#!/usr/bin/env python

import os

from datetime import datetime

from rnn.seq2seq_autoencoder import LSTMAutoencoder, LSTMEncoder, Classifier
from dtw.data import Data
from dtw.dtw import DTW

from keras.preprocessing import sequence

PATH = os.path.dirname(__file__)

model_save_template = os.path.join(PATH, '{name}_model.dat')
evaluation_save_template = os.path.join(PATH, '{timestamp}_evaluation.dat')

# DTW Configuration
window_size = 4

# Data Configuration
user_count = 11
genuine_sample_count = 42
forged_sample_count = 36
forger_count = 4
train_slice_per = 80
genuine_file_path_template = os.path.join(PATH, 'data/Genuine/{user}/{sample}_{user}.HWR')
forged_file_path_template = os.path.join(PATH, 'data/Forged/{user}/{sample}_{forger}_{user}.HWR')

# Auto encoder Configuration
encoded_length = 100
input_dim = 2
ae_nb_epoch = 100

# Classifier Configuration
c_nb_epoch = 100


def generate_data():
    data = Data(
        user_count=user_count,
        genuine_sample_count=genuine_sample_count,
        forged_sample_count=forged_sample_count,
        forger_count=forger_count,
        genuine_file_path_template=genuine_file_path_template,
        forged_file_path_template=forged_file_path_template,
        train_slice_per=train_slice_per
    )
    return data


def run_dtw(data):
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


def get_auto_encoder(x, max_len):
    ae = LSTMAutoencoder(input_length=max_len, input_dimension=input_dim, encoded_length=encoded_length)
    ae.fit(train_input=x, nb_epoch=ae_nb_epoch)
    ae.save(save_path=model_save_template.format(name='auto_encoder'))
    return ae


def get_encoder(max_len):
    e = LSTMEncoder(input_length=max_len, input_dimension=input_dim, encoded_length=encoded_length)
    e.load(load_path=model_save_template.format(name='auto_encoder'))
    return e


def get_classifier(x, y):
    c = Classifier(encoded_length=encoded_length)
    c.fit(train_x=x, train_y=y, nb_epoch=c_nb_epoch)
    c.save(save_path=model_save_template.format(name='classifier'))
    return c


def generate_padded_input_data(x, max_len):
    return sequence.pad_sequences(x, maxlen=max_len)


def train():
    data = generate_data()

    train_max_len = data.train_max_len
    x = generate_padded_input_data(data.train_x, train_max_len)
    y = data.train_y

    get_auto_encoder(x, train_max_len)
    encoder = get_encoder(train_max_len)
    encoded_x = encoder.predict(x)

    dev_max_len = data.dev_max_len
    dev_x = generate_padded_input_data(data.dev_x, dev_max_len)
    dev_y = data.dev_y

    return get_classifier(encoded_x, y), encoder.predict(dev_x), dev_y


def dev(classifier, x, y):
    with open(evaluation_save_template.format(timestamp=int(datetime.now().timestamp())), 'a') as f:
        metrics = classifier.evaluate(x, y)
        for _ in metrics:
            f.write(str(_) + '\n')


if __name__ == '__main__':
    dev(*train())
