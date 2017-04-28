#!/usr/bin/env python

import csv
import errno
import json
import logging
import os

import numpy as np
from keras import layers, optimizers, losses, metrics
from keras.preprocessing import sequence
from sklearn.metrics import classification_report

from data import Data
from seq2seq.models import Autoencoder, Encoder, LinearSVC

PATH = os.path.dirname(os.path.abspath(__file__))
CONIFG_PATH = os.path.join(PATH, 'configuration.json')
if not os.path.exists(CONIFG_PATH):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'configuration.json')
with open(CONIFG_PATH, 'r') as cf:
    CONFIG = json.load(cf)

# General Configuration
verbose = CONFIG['general']['verbose']
implementation = CONFIG['general']['implementation']

# Export Configuration
mdl_save_temp = os.path.join(PATH, CONFIG['export']['model_save_template'])
csv_fns = CONFIG['export']['csv_fieldnames']

# Data Configuration
smp_stp = CONFIG['data']['sampling_step']
rl_win_sz = CONFIG['data']['rolling_window_size']
rl_win_stp = CONFIG['data']['rolling_window_step']
nrm = CONFIG['data']['normalization']
usr_cnt = CONFIG['data']['user_count']
inp_dim = CONFIG['data']['input_dimension']
gen_smp_cnt = CONFIG['data']['genuine_sample_count']
frg_smp_cnt = CONFIG['data']['forged_sample_count']
gen_path_temp = os.path.join(PATH, CONFIG['data']['genuine_path_template'])
frg_path_temp = os.path.join(PATH, CONFIG['data']['forged_path_template'])
ftr_cnt = CONFIG['data']['feature_count']

# Autoencoder Configuration
mask_value = CONFIG['autoencoder']['mask_value']
ae_btch_sz = CONFIG['autoencoder']['batch_size']
enc_arc = CONFIG['autoencoder']['encoder_architecture']
dec_arc = CONFIG['autoencoder']['decoder_architecture']
ae_tr_epochs = CONFIG['autoencoder']['train_epochs']
cell_type = CONFIG['autoencoder']['cell_type']
ae_loss = getattr(losses, CONFIG['autoencoder']['loss'])
ae_optimizer = getattr(optimizers, CONFIG['autoencoder']['optimizer']['name'])(
    **CONFIG['autoencoder']['optimizer']['args']
)
ae_metrics = [getattr(metrics, _) if hasattr(metrics, _) else _ for _ in CONFIG['autoencoder']['metrics']]

# Logger Configuration
log_frm = CONFIG['logger']['log_format']
log_fl = CONFIG['logger']['log_file'].format(
    ct=cell_type,
    earc='x'.join(map(str, enc_arc)),
    darc='x'.join(map(str, dec_arc)),
    epc=ae_tr_epochs
)
if not os.path.exists(os.path.dirname(log_fl)):
    os.mkdir(os.path.dirname(log_fl))
log_lvl = getattr(logging, CONFIG['logger']['log_level'].upper())
logging.basicConfig(filename=log_fl, level=log_lvl, format=log_frm)
logger = logging.getLogger(__name__)


def get_data():
    return Data(
        smp_stp=smp_stp,
        rl_win_sz=rl_win_sz,
        rl_win_stp=rl_win_stp,
        ftr_cnt=ftr_cnt,
        nrm=nrm,
        usr_cnt=usr_cnt,
        gen_smp_cnt=gen_smp_cnt,
        frg_smp_cnt=frg_smp_cnt,
        gen_path_temp=gen_path_temp,
        frg_path_temp=frg_path_temp
    )


def train_autoencoder(x, y, btch, epc, earc, darc, ct, usr_num, msk_val):
    logger.info('Training Autoencoder for user {usr_num}'.format(usr_num=usr_num))
    cell = getattr(layers, ct)
    ae = Autoencoder(
        cell=cell,
        inp_dim=inp_dim,
        max_len=x.shape[1],
        earc=earc,
        darc=darc,
        loss=ae_loss,
        optimizer=ae_optimizer,
        metrics=ae_metrics,
        implementation=implementation,
        mask_value=msk_val
    )
    ae.fit(x, y, epochs=epc, batch_size=btch, verbose=verbose)
    ae.save(path=mdl_save_temp.format(name='models/{usr_num}_{ct}_autoencoder_{earc}_{darc}_{epc}'.format(
        usr_num=usr_num, ct=ct, earc='x'.join(map(str, earc)), darc='x'.join(map(str, darc)), epc=epc
    )))


def load_encoder(x, y, btch, epc, earc, darc, ct, usr_num, msk_val):
    train_autoencoder(x, y, btch, epc, earc, darc, ct, usr_num, msk_val)

    cell = getattr(layers, ct)
    e = Encoder(
        cell=cell,
        inp_dim=inp_dim,
        earc=earc,
        loss=ae_loss,
        optimizer=ae_optimizer,
        metrics=ae_metrics,
        implementation=implementation,
        mask_value=msk_val
    )
    e.load(path=mdl_save_temp.format(name='models/{usr_num}_{ct}_autoencoder_{earc}_{darc}_{epc}'.format(
        usr_num=usr_num, ct=ct, earc='x'.join(map(str, earc)), darc='x'.join(map(str, darc)), epc=epc
    )))
    return e


def save_evaluations(evls, fns, epc, earc, darc, ct):
    dir_path = os.path.join(PATH, 'models/{ct}-{earc}-{darc}-{epc}'.format(
        ct=ct, earc='x'.join(map(str, earc)), darc='x'.join(map(str, darc)), epc=epc
    ))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    with open(os.path.join(dir_path, 'evaluation.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(evls)
        avg = {
            fns[0]: 'AVG'
        }
        avg.update({
            fns[i]: np.mean([evl[fns[i]] for evl in evls]) for i in range(1, len(fns))
        })
        w.writerow(avg)


def train_lsvc(x, y, usr_num, earc, darc, fns):
    c = LinearSVC()
    c.fit(x, y)
    c.save(path=mdl_save_temp.format(name='models/{usr_num}_lsvc_{earc}_{darc}'.format(
        usr_num=usr_num, earc='x'.join(map(str, earc)), darc='x'.join(map(str, darc))
    )))
    cr = list(map(float, classification_report(y_true=y, y_pred=c.predict(x)).split('\n')[-2].split()[3:6]))
    return {
        fns[0]: usr_num,
        fns[3]: cr[0],
        fns[4]: cr[1],
        fns[5]: cr[2]
    }


def get_lsvc_train_data(enc_gen, enc_frg):
    return np.concatenate((enc_gen, enc_frg)), \
           np.concatenate((np.ones_like(enc_gen[:, 0]), np.zeros_like(enc_frg[:, 0])))


def evaluate_model(usr_num, enc_gen, enc_frg, earc, darc, fns):
    x, y = get_lsvc_train_data(enc_gen, enc_frg)
    return train_lsvc(x, y, usr_num, earc, darc, fns)


def get_encoded_data(e, gen_x, frg_x, msk_val):
    enc_gen = e.predict(inp=sequence.pad_sequences(gen_x, value=msk_val))  # Encoded Genuine Data
    enc_frg = e.predict(inp=sequence.pad_sequences(frg_x, value=msk_val))  # Encoded Forged Data
    return enc_gen, enc_frg


def get_autoencoder_train_data(data, usr_num, msk_val):
    (gen_x, gen_y) = data.get_genuine_combinations(usr_num)
    x, y = sequence.pad_sequences(gen_x, value=msk_val), sequence.pad_sequences(gen_y, value=msk_val)
    return x, y, sequence.pad_sequences(data.gen[usr_num], value=msk_val), \
        sequence.pad_sequences(data.frg[usr_num], value=msk_val)


def process_models(data, btch, epc, earc, darc, ct, msk_val, fns):
    evls = list()
    for usr_num in range(usr_cnt):
        x, y, gen_x, frg_x = get_autoencoder_train_data(data, usr_num, msk_val)
        e = load_encoder(x, y, btch, epc, earc, darc, ct, usr_num, msk_val)
        enc_gen, enc_frg = get_encoded_data(e, gen_x, frg_x, msk_val)
        evl = evaluate_model(usr_num, enc_gen, enc_frg, earc, darc, fns)
        evl.update({
            fns[1]: np.mean(list(map(len, data.gen[usr_num]))),
            fns[2]: np.mean(list(map(len, data.frg[usr_num])))
        })
        evls.append(evl)
    save_evaluations(evls, fns, epc, earc, darc, ct)

if __name__ == '__main__':
    process_models(get_data(), ae_btch_sz, ae_tr_epochs, enc_arc, dec_arc, cell_type, mask_value, csv_fns)
