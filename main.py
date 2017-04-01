#!/usr/bin/env python

import errno
import itertools
import json
import logging
import os

import numpy as np
from keras.layers import LSTM, GRU
from keras.preprocessing import sequence

from data import Data
from rnn import Autoencoder, Encoder

PATH = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(PATH, 'configuration.json'), 'r') as cf:
        CONFIG = json.load(cf)
except json.JSONDecodeError:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'configuration.json')

# Logger Configuration
log_frm = CONFIG['logger']['log_format']
log_fl = CONFIG['logger']['log_file']
log_lvl = getattr(logging, CONFIG['logger']['log_level'].upper())
logging.basicConfig(filename=log_fl, level=log_lvl, format=log_frm)
logger = logging.getLogger(__name__)

# Export Configuration
mdl_save_temp = os.path.join(PATH, CONFIG['export']['model_save_template'])

# Data Configuration
usr_cnt = CONFIG['data']['user_count']
inp_dim = CONFIG['data']['input_dimension']
gen_smp_cnt = CONFIG['data']['genuine_sample_count']
frg_smp_cnt = CONFIG['data']['forged_sample_count']
frg_cnt = CONFIG['data']['forger_count']
gen_path_temp = os.path.join(PATH, CONFIG['data']['genuine_path_template'])
frg_path_temp = os.path.join(PATH, CONFIG['data']['forged_path_template'])

# Autoencoder Configuration
btch_sz = CONFIG['autoencoder']['batch_size']
enc_lens_str = CONFIG['autoencoder']['encoded_length']['start']
enc_lens_fns = CONFIG['autoencoder']['encoded_length']['finish']
enc_lens_stp = CONFIG['autoencoder']['encoded_length']['step']
enc_lens = range(enc_lens_str, enc_lens_fns + 1, enc_lens_stp)
tr_epochs_str = CONFIG['autoencoder']['train_epochs']['start']
tr_epochs_fns = CONFIG['autoencoder']['train_epochs']['finish']
tr_epochs_stp = CONFIG['autoencoder']['train_epochs']['step']
tr_epochs = range(tr_epochs_str, tr_epochs_fns + 1, tr_epochs_stp)
cell_types = CONFIG['autoencoder']['cell_types']


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


def train_autoencoder(x, y, max_len, btch, epc, el, ct, usr_num):
    logger.info('Training Autoencoder for user {usr_num}'.format(usr_num=usr_num))
    cell = LSTM if ct == 'lstm' else GRU
    ae = Autoencoder(cell=cell, inp_max_len=max_len, inp_dim=inp_dim, enc_len=el)
    ae.fit(x, y, epochs=epc, batch_size=btch)
    ae.save(path=mdl_save_temp.format(name='{usr_num}_{ct}_autoencoder_{el}_{epc}'.format(
        usr_num=usr_num, ct=ct, el=el, epc=epc))
    )
    return ae


def load_encoder(max_len, epc, el, ct, usr_num):
    logger.info('Loading Encoder for user {usr_num}'.format(usr_num=usr_num))
    cell = LSTM if ct == 'lstm' else GRU
    e = Encoder(cell=cell, inp_max_len=max_len, inp_dim=inp_dim, enc_len=el)
    e.load(path=mdl_save_temp.format(name='{usr_num}_{ct}_autoencoder_{el}_{epc}'.format(
        usr_num=usr_num, ct=ct, el=el, epc=epc))
    )
    return e


def pad_sequence(x, max_len=None):
    return sequence.pad_sequences(x, maxlen=max_len)


def get_encoded_data(data, btch, epc, el, ct):
    enc_gen, enc_frg = list(), list()
    for usr_num in range(usr_cnt):
        (gen_x, gen_y), (frg_x, frg_y) = data.get_combinations(usr_num, forged=False), \
                                         data.get_combinations(usr_num, forged=True)
        x, y = pad_sequence(gen_x + frg_x), pad_sequence(gen_y + frg_y)

        max_len = x.shape[1]
        train_autoencoder(x, y, max_len, btch, epc, el, ct, usr_num)  # Autoencoder
        e = load_encoder(max_len, epc, el, ct, usr_num)  # Encoder

        logger.info('Encoding Data for user {usr_num}'.format(usr_num=usr_num))
        enc_gen.append(e.predict(inp=pad_sequence(gen_x, max_len), batch_size=btch))  # Encoded Genuine Data
        enc_frg.append(e.predict(inp=pad_sequence(frg_x, max_len), batch_size=btch))  # Encoded Forged Data

    return enc_gen, enc_frg


def save_encoded_distances(gen, frg, epc, el, ct):
    dir_path = os.path.join(PATH, 'models/{ct}-{el}-{epc}'.format(ct=ct, el=el, epc=epc))
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


if __name__ == '__main__':
    d = get_data()
    for epochs in tr_epochs:
        for cell_type, enc_len in itertools.product(cell_types, enc_lens):
            logger.info('Started, cell type is \'{cell_type}\', encoded length is \'{enc_len}\''.format(
                cell_type=cell_type, enc_len=enc_len)
            )
            e_gen, e_frg = get_encoded_data(d, btch_sz, epochs, enc_len, cell_type)
            save_encoded_distances(e_gen, e_frg, epochs, enc_len, cell_type)
            logger.info('Finished with {epochs} epochs!'.format(epochs=epochs))
