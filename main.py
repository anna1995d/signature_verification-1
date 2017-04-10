#!/usr/bin/env python

import errno
import itertools
import json
import logging
import os

import numpy as np
from keras import layers, optimizers, losses, metrics
from keras.preprocessing import sequence

from data import Data
from nn import Autoencoder, Encoder, Classifier

PATH = os.path.dirname(os.path.abspath(__file__))
CONIFG_PATH = os.path.join(PATH, 'configuration.json')
if not os.path.exists(CONIFG_PATH):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'configuration.json')
with open(CONIFG_PATH, 'r') as cf:
    CONFIG = json.load(cf)

# General Configuration
verbose = CONFIG['general']['verbose']
implementation = CONFIG['general']['implementation']

# Logger Configuration
log_frm = CONFIG['logger']['log_format']
log_fl = CONFIG['logger']['log_file']
log_lvl = getattr(logging, CONFIG['logger']['log_level'].upper())
logging.basicConfig(filename=log_fl, level=log_lvl, format=log_frm)
logger = logging.getLogger(__name__)

# Export Configuration
mdl_save_temp = os.path.join(PATH, CONFIG['export']['model_save_template'])

# Data Configuration
smp_stp = CONFIG['data']['sampling_step']
usr_cnt = CONFIG['data']['user_count']
inp_dim = CONFIG['data']['input_dimension']
gen_smp_cnt = CONFIG['data']['genuine_sample_count']
frg_smp_cnt = CONFIG['data']['forged_sample_count']
frg_cnt = CONFIG['data']['forger_count']
gen_path_temp = os.path.join(PATH, CONFIG['data']['genuine_path_template'])
frg_path_temp = os.path.join(PATH, CONFIG['data']['forged_path_template'])

# Autoencoder Configuration
with_forged = CONFIG['autoencoder']['with_forged']
mask_value = CONFIG['autoencoder']['mask_value']
ae_btch_sz = CONFIG['autoencoder']['batch_size']
enc_lens_str = CONFIG['autoencoder']['encoded_length']['start']
enc_lens_fns = CONFIG['autoencoder']['encoded_length']['finish']
enc_lens_stp = CONFIG['autoencoder']['encoded_length']['step']
enc_lens = range(enc_lens_str, enc_lens_fns + 1, enc_lens_stp)
ae_tr_epochs = CONFIG['autoencoder']['train_epochs']
cell_types = CONFIG['autoencoder']['cell_types']
ae_loss = getattr(losses, CONFIG['autoencoder']['loss'])
ae_optimizer = getattr(optimizers, CONFIG['autoencoder']['optimizer']['name'])(
    **CONFIG['autoencoder']['optimizer']['args']
)
ae_metrics = [getattr(metrics, _) if hasattr(metrics, _) else _ for _ in CONFIG['autoencoder']['metrics']]

# Classifier Configuration
cf_btch_sz = CONFIG['classifier']['batch_size']
cf_tr_epochs = CONFIG['classifier']['train_epochs']
cf_activation = CONFIG['classifier']['activation']
cf_loss = getattr(losses, CONFIG['classifier']['loss'])
cf_optimizer = getattr(optimizers, CONFIG['classifier']['optimizer']['name'])(
    **CONFIG['classifier']['optimizer']['args']
)
cf_metrics = [getattr(metrics, _) if hasattr(metrics, _) else _ for _ in CONFIG['classifier']['metrics']]


def get_data():
    logger.info('Getting Data')
    return Data(
        smp_stp=smp_stp,
        usr_cnt=usr_cnt,
        gen_smp_cnt=gen_smp_cnt,
        frg_smp_cnt=frg_smp_cnt,
        frg_cnt=frg_cnt,
        gen_path_temp=gen_path_temp,
        frg_path_temp=frg_path_temp,
    )


def train_autoencoder(x, y, max_len, btch, epc, el, ct, usr_num, msk_val):
    logger.info('Training Autoencoder for user {usr_num}'.format(usr_num=usr_num))
    cell = getattr(layers, ct)
    ae = Autoencoder(
        cell=cell,
        inp_max_len=max_len,
        inp_dim=inp_dim,
        enc_len=el,
        loss=ae_loss,
        optimizer=ae_optimizer,
        metrics=ae_metrics,
        implementation=implementation,
        mask_value=msk_val
    )
    ae.fit(x, y, epochs=epc, batch_size=btch, verbose=verbose)
    ae.save(path=mdl_save_temp.format(name='models/{usr_num}_{ct}_autoencoder_{el}_{epc}'.format(
        usr_num=usr_num, ct=ct, el=el, epc=epc
    )))


def load_encoder(x, y, btch, max_len, epc, el, ct, usr_num, msk_val):
    train_autoencoder(x, y, max_len, btch, epc, el, ct, usr_num, msk_val)

    logger.info('Loading Encoder for user {usr_num}'.format(usr_num=usr_num))
    cell = getattr(layers, ct)
    e = Encoder(
        cell=cell,
        inp_max_len=max_len,
        inp_dim=inp_dim,
        enc_len=el,
        loss=ae_loss,
        optimizer=ae_optimizer,
        metrics=ae_metrics,
        implementation=implementation,
        mask_value=msk_val
    )
    e.load(path=mdl_save_temp.format(name='models/{usr_num}_{ct}_autoencoder_{el}_{epc}'.format(
        usr_num=usr_num, ct=ct, el=el, epc=epc
    )))
    return e


def pad_sequence(x, msk_val, max_len=None):
    return sequence.pad_sequences(x, value=msk_val, maxlen=max_len)


def save_evaluation(h, usr_num, epc, el, ct):
    dir_path = os.path.join(PATH, 'models/{ct}-{el}-{epc}'.format(ct=ct, el=el, epc=epc))

    with open(os.path.join(dir_path, 'evaluation.txt'), 'a') as f:
        f.write('User {usr_num}: acc is {acc}, loss is {loss}\n'.format(
            usr_num=usr_num, acc=h.history['acc'][-1], loss=h.history['loss'][-1]
        ))


def train_classifier(x, y, usr_num, epc, el):
    c = Classifier(
        activation=cf_activation,
        loss=cf_loss,
        optimizer=cf_optimizer,
        metrics=cf_metrics,
    )
    h = c.fit(x, y, epochs=epc, batch_size=cf_btch_sz, verbose=verbose)
    c.save(path=mdl_save_temp.format(name='models/{usr_num}_classifier_{el}_{epc}'.format(
        usr_num=usr_num, el=el, epc=epc
    )))
    return h


def get_classifier_train_data(usr_num, epc, el, ct):
    dir_path = os.path.join(PATH, 'models/{ct}-{el}-{epc}'.format(ct=ct, el=el, epc=epc))

    with open(os.path.join(dir_path, 'encoded_genuine_{usr_num}.txt'.format(usr_num=usr_num)), 'r') as f:
        x_gen = list(map(lambda x: float(x), f.read().split()))

    with open(os.path.join(dir_path, 'encoded_genuine_forged_{usr_num}.txt'.format(usr_num=usr_num)), 'r') as f:
        x_frg = list(map(lambda x: float(x), f.read().split()))

    return np.concatenate((x_gen, x_frg)), np.concatenate((np.ones_like(x_gen), np.zeros_like(x_frg)))


def evaluate_model(usr_num, el, ct):
    x, y = get_classifier_train_data(usr_num, ae_tr_epochs, el, ct)
    h = train_classifier(x, y, usr_num, cf_tr_epochs, el)
    save_evaluation(h, usr_num, ae_tr_epochs, el, ct)


def save_encoded_distances(usr, gen, frg, epc, el, ct):
    dir_path = os.path.join(PATH, 'models/{ct}-{el}-{epc}'.format(ct=ct, el=el, epc=epc))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    logger.info('Saving Encoded Distance: {file}'.format(file='encoded_genuine_{usr}.txt'.format(usr=usr)))
    with open(os.path.join(dir_path, 'encoded_genuine_{usr}.txt'.format(usr=usr)), 'a') as f:
        for x, y in itertools.combinations(gen, 2):
            f.write(str(np.linalg.norm(x - y)) + '\n')

    logger.info('Saving Encoded Distance: {file}'.format(file='encoded_genuine_forged_{usr}.txt'.format(usr=usr)))
    with open(os.path.join(dir_path, 'encoded_genuine_forged_{usr}.txt'.format(usr=usr)), 'a') as f:
        for x, y in itertools.product(gen, frg):
            f.write(str(np.linalg.norm(x - y)) + '\n')


def get_encoded_data(usr_num, e, btch, gen_x, frg_x, max_len, msk_val):
    logger.info('Encoding Data for user {usr_num}'.format(usr_num=usr_num))
    enc_gen = e.predict(inp=pad_sequence(gen_x, msk_val, max_len), batch_size=btch)  # Encoded Genuine Data
    enc_frg = e.predict(inp=pad_sequence(frg_x, msk_val, max_len), batch_size=btch)  # Encoded Forged Data
    return enc_gen, enc_frg


def get_autoencoder_train_data(data, usr_num, with_frg, msk_val):
    (gen_x, gen_y), (frg_x, frg_y) = data.get_combinations(usr_num, forged=False), \
                                     data.get_combinations(usr_num, forged=True)
    x, y = pad_sequence((gen_x + frg_x) if with_frg else gen_x, msk_val), \
        pad_sequence((gen_y + frg_y) if with_frg else gen_y, msk_val)
    max_len = x.shape[1]
    return x, y, pad_sequence(data.gen[usr_num], msk_val, max_len), pad_sequence(data.frg[usr_num], msk_val, max_len), \
        max_len


def process_models(data, btch, epc, el, ct, with_frg, msk_val):
    for usr_num in range(usr_cnt):
        x, y, gen_x, frg_x, max_len = get_autoencoder_train_data(data, usr_num, with_frg, msk_val)
        e = load_encoder(x, y, btch, max_len, epc, el, ct, usr_num, msk_val)
        enc_gen, enc_frg = get_encoded_data(usr_num, e, btch, gen_x, frg_x, max_len, msk_val)
        save_encoded_distances(usr_num, enc_gen, enc_frg, epc, el, ct)
        evaluate_model(usr_num, enc_len, cell_type)


if __name__ == '__main__':
    d = get_data()
    for cell_type, enc_len in itertools.product(cell_types, enc_lens):
        logger.info('Started, cell type is \'{cell_type}\', encoded length is \'{enc_len}\''.format(
            cell_type=cell_type, enc_len=enc_len
        ))
        process_models(d, ae_btch_sz, ae_tr_epochs, enc_len, cell_type, with_forged, mask_value)
        logger.info('Finished with {epochs} epochs!'.format(epochs=ae_tr_epochs))
