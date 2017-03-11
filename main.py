#!/usr/bin/env python

import itertools
import os

import numpy as np
from keras.preprocessing import sequence

from data.data import Data
from dtw.dtw import DTW
from rnn.seq2seq import LSTMAutoEncoder, LSTMEncoder

PATH = os.path.dirname(__file__)

# Export Configuration
mdl_save_temp = os.path.join(PATH, '{name}_model.dat')
eval_save_temp = os.path.join(PATH, '{timestamp}_evaluation.dat')

# DTW Configuration
win_len = 4

# Data Configuration
usr_cnt = 11
gen_smp_cnt = 42
frg_smp_cnt = 36
frg_cnt = 4
gen_path_temp = os.path.join(PATH, 'data_set/Genuine/{user}/{sample}_{user}.HWR')
frg_path_temp = os.path.join(PATH, 'data_set/Forged/{user}/{sample}_{forger}_{user}.HWR')

# Auto encoder Configuration
enc_len = 200
inp_dim = 2
ae_nb_epoch = 200


def get_data():
    return Data(
        usr_cnt=usr_cnt,
        gen_smp_cnt=gen_smp_cnt,
        frg_smp_cnt=frg_smp_cnt,
        frg_cnt=frg_cnt,
        gen_path_temp=gen_path_temp,
        frg_path_temp=frg_path_temp,
    )


def train_auto_encoder(x, max_len):
    ae = LSTMAutoEncoder(inp_max_len=max_len, inp_dim=inp_dim, enc_len=enc_len)
    ae.fit(tr_inp=x, nb_epoch=ae_nb_epoch)
    ae.save(path=mdl_save_temp.format(name='auto_encoder'))
    return ae


def load_encoder(max_len):
    e = LSTMEncoder(inp_max_len=max_len, inp_dim=inp_dim, enc_len=enc_len)
    e.load(path=mdl_save_temp.format(name='auto_encoder'))
    return e


def pad_sequence(x, max_len=None):
    return sequence.pad_sequences(x, maxlen=max_len)


def get_encoded_data(data):
    x = pad_sequence(data.train)

    ae = train_auto_encoder(x, data.max_len)  # Auto Encoder

    e = load_encoder(data.max_len)  # Encoder

    enc_gen = [e.predict(pad_sequence(gen, data.max_len)) for gen in d.gen]  # Encoded Genuine Data
    enc_frg = [e.predict(pad_sequence(frg, data.max_len)) for frg in d.frg]  # Encoded Forged Data

    return enc_gen, enc_frg


def save_dtw_distances(data):
    with open(PATH + 'dtw_genuine.txt', 'a') as f:
        for usr in range(usr_cnt):
            for x, y in data.get_combinations(usr, forged=False):
                dtw = DTW(x, y, win_len, DTW.euclidean)
                f.write(str(dtw.calculate()) + '\n')

    with open(PATH + 'dtw_genuine_forged.txt', 'a') as f:
        for usr in range(usr_cnt):
            for x, y in data.get_combinations(usr, forged=True):
                dtw = DTW(x, y, win_len, DTW.euclidean)
                f.write(str(dtw.calculate()) + '\n')


def save_encoded_distances(gen, frg):
    for usr in range(usr_cnt):
        with open(PATH + 'encoded_genuine_{usr}.txt'.format(usr=usr), 'a') as f:
            for x, y in itertools.combinations(gen[usr], 2):
                f.write(str(np.linalg.norm(x - y)) + '\n')

    for usr in range(usr_cnt):
        with open(PATH + 'encoded_genuine_forged_{usr}.txt'.format(usr=usr), 'a') as f:
            for x, y in itertools.product(gen[usr], frg[usr]):
                f.write(str(np.linalg.norm(x - y)) + '\n')


def save_dtw_threshold():
    with open(PATH + 'dtw_genuine.txt', 'r') as f:
        px = np.sort(np.array(f.read().split('\n')[:-1]).astype(np.float))

    with open(PATH + 'dtw_genuine_forged.txt', 'r') as f:
        nx = np.sort(np.array(f.read().split('\n')[:-1]).astype(np.float))

    ax = itertools.chain(px, nx)
    mini = min(enumerate(ax), key=lambda x: np.linalg.norm(px[px < x[1]] - x[1]) + np.linalg.norm(nx[nx > x[1]] - x[1]))
    mind = np.linalg.norm(px[px < mini[1]] - mini[1]) + np.linalg.norm(nx[nx > mini[1]] - mini[1])

    with open(PATH + 'dtw_threshold.txt', 'w') as f:
        f.write('Threshold: {t}\nMin Distance: {d}\n'.format(t=str(mini[1]), d=str(mind)))


if __name__ == '__main__':
    d = get_data()
    e_gen, e_frg = get_encoded_data(d)

    save_encoded_distances(e_gen, e_frg)
    save_dtw_distances(d)
