#!/usr/bin/env python

import itertools
import logging
import os

import numpy as np
from keras.layers import LSTM, GRU
from keras.preprocessing import sequence

from data.data import Data
from dtw.dtw import DTW
from rnn.seq2seq import AutoEncoder, Encoder

PATH = os.path.dirname(__file__)

logging.basicConfig(filename='seq2seq.log', level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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
enc_len = 100
inp_dim = 2
ae_nb_epoch = 10
cell_type = 'lstm'


def get_data():
    logger.info('Getting Data')
    return Data(
        usr_cnt=usr_cnt,
        gen_smp_cnt=gen_smp_cnt,
        frg_smp_cnt=frg_smp_cnt,
        frg_cnt=frg_cnt,
        gen_path_temp=gen_path_temp,
        frg_path_temp=frg_path_temp,
    )


def train_auto_encoder(x, max_len, ct):
    logger.info('Training Auto Encoder')
    cell = LSTM if ct == 'lstm' else GRU
    ae = AutoEncoder(cell=cell, inp_max_len=max_len, inp_dim=inp_dim, enc_len=enc_len)
    ae.fit(tr_inp=x, nb_epoch=ae_nb_epoch)
    ae.save(path=mdl_save_temp.format(name='{cell_type}_auto_encoder_{enc_len}_{epoch}'.format(
        cell_type=ct, enc_len=enc_len, epoch=ae_nb_epoch
    )))
    return ae


def load_encoder(max_len, ct):
    logger.info('Loading Encoder')
    cell = LSTM if ct == 'lstm' else GRU
    e = Encoder(cell=cell, inp_max_len=max_len, inp_dim=inp_dim, enc_len=enc_len)
    e.load(path=mdl_save_temp.format(name='{cell_type}_auto_encoder_{enc_len}_{epoch}'.format(
        cell_type=ct, enc_len=enc_len, epoch=ae_nb_epoch
    )))
    return e


def pad_sequence(x, max_len=None):
    logger.info('Padding Sequences')
    return sequence.pad_sequences(x, maxlen=max_len)


def get_encoded_data(data, ct):
    x = pad_sequence(data.train)

    train_auto_encoder(x, data.max_len, ct)  # Auto Encoder
    e = load_encoder(data.max_len, ct)  # Encoder

    logger.info('Encoding Data')
    enc_gen = [e.predict(pad_sequence(gen, data.max_len)) for gen in d.gen]  # Encoded Genuine Data
    enc_frg = [e.predict(pad_sequence(frg, data.max_len)) for frg in d.frg]  # Encoded Forged Data

    return enc_gen, enc_frg


def save_dtw_distances(data):
    logger.info('Saving DTW Distances')
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


def save_encoded_distances(gen, frg, ct):
    dir_path = os.path.join(PATH, 'models/{ct}-{enc_len}-{epoch}'.format(ct=ct, enc_len=enc_len, epoch=ae_nb_epoch))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for usr in range(usr_cnt):
        logger.info('Saving Encoded Distance: {file}'.format(file='encoded_genuine_{usr}.txt'.format(usr=usr)))
        with open(os.path.join(dir_path, 'encoded_genuine_{usr}.txt'.format(usr=usr)), 'a') as f:
            for x, y in itertools.combinations(gen[usr], 2):
                f.write(str(np.linalg.norm(x - y)) + '\n')

    for usr in range(usr_cnt):
        logger.info('Saving Encoded Distance: {file}'.format(file='encoded_genuine_forged_{usr}.txt'.format(usr=usr)))
        with open(os.path.join(dir_path, 'encoded_genuine_forged_{usr}.txt'.format(usr=usr)), 'a') as f:
            for x, y in itertools.product(gen[usr], frg[usr]):
                f.write(str(np.linalg.norm(x - y)) + '\n')


def save_dtw_threshold():
    logger.info('Saving DTW Threshold')
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
    for cell_type, enc_len in itertools.product(['lstm', 'gru'], range(100, 1001, 100)):
        logger.info('Started, cell type is \'{cell_type}\', encoded length is \'{enc_len}\''.format(
            cell_type=cell_type, enc_len=enc_len)
        )
        e_gen, e_frg = get_encoded_data(d, cell_type)
        save_encoded_distances(e_gen, e_frg, cell_type)
        logger.info('Finished!')
    save_dtw_distances(d)
