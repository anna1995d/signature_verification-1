#!/usr/bin/env python

import csv
import errno
import json
import logging
import os

import numpy as np
from keras import layers, optimizers
from keras.preprocessing import sequence
from sklearn.metrics import classification_report

from data import Data
from seq2seq.rnn.models import Autoencoder, Encoder, LinearSVC

PATH = os.path.dirname(os.path.abspath(__file__))
CONIFG_PATH = os.path.join(PATH, 'configuration.json')
if not os.path.exists(CONIFG_PATH):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'configuration.json')
with open(CONIFG_PATH, 'r') as cf:
    CONFIG = json.load(cf)

# General Configuration
verbose = CONFIG['general']['verbose']
output_directory_temp = os.path.join(PATH, CONFIG['general']['output_directory_template'])

# Export Configuration
mdl_save_temp = CONFIG['export']['model_save_template']
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
mask_value = CONFIG['rnn']['autoencoder']['mask_value']
ae_btch_sz = CONFIG['rnn']['autoencoder']['batch_size']
enc_arc = CONFIG['rnn']['autoencoder']['encoder_architecture']
dec_arc = CONFIG['rnn']['autoencoder']['decoder_architecture']
ae_tr_epochs = CONFIG['rnn']['autoencoder']['train_epochs']
cell_type = CONFIG['rnn']['autoencoder']['cell_type']
bd_cell_type = CONFIG['rnn']['autoencoder']['bidirectional']
bd_merge_mode = CONFIG['rnn']['autoencoder']['bidirectional_merge_mode']
ae_ccfg = CONFIG['rnn']['autoencoder']['compile_config']
ae_ccfg['optimizer'] = getattr(optimizers, ae_ccfg['optimizer']['name'])(**ae_ccfg['optimizer']['args'])
ae_lcfg = CONFIG['rnn']['autoencoder']['layers_config']

# Logger Configuration
log_frm = CONFIG['logger']['log_format']
log_fl = CONFIG['logger']['log_file'].format(
    bd='b' if bd_cell_type else '',
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


def train_autoencoder(x, y, epc, earc, darc, ct, bd, usr_num, msk_val, aes_dir):
    logger.info('Training Autoencoder for user {usr_num}'.format(usr_num=usr_num))
    cell = getattr(layers, ct)
    ae = Autoencoder(
        cell=cell,
        bidir=bd,
        bidir_mrgm=bd_merge_mode,
        inp_dim=inp_dim,
        max_len=x.shape[1],
        earc=earc,
        darc=darc,
        msk_val=msk_val,
        ccfg=ae_ccfg,
        lcfg=ae_lcfg
    )
    ae.fit(x, y, epochs=epc, batch_size=ae_btch_sz, verbose=verbose, usr_num=usr_num)
    ae.save(path=os.path.join(aes_dir, mdl_save_temp.format(usr_num=usr_num)))


def load_encoder(x, y, epc, earc, darc, ct, bd, usr_num, msk_val, aes_dir):
    train_autoencoder(x, y, epc, earc, darc, ct, bd, usr_num, msk_val, aes_dir)

    cell = getattr(layers, ct)
    e = Encoder(
        cell=cell,
        bidir=bd,
        bidir_mrgm=bd_merge_mode,
        inp_dim=inp_dim,
        earc=earc,
        msk_val=msk_val,
        ccfg=ae_ccfg,
        lcfg=ae_lcfg
    )
    e.load(path=os.path.join(aes_dir, mdl_save_temp.format(usr_num=usr_num)))
    return e


def save_evaluation(evl, fns, out_dir):
    with open(os.path.join(out_dir, 'evaluations.csv'), 'a') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writerow(evl)


def save_avg_evaluation(fns, out_dir):
    with open(os.path.join(out_dir, 'evaluations.csv'), 'r') as f:
        avg = {
            fns[0]: 'AVG'
        }
        rows = [r for r in csv.DictReader(f, fieldnames=fns)][1:]
        avg.update({
            fns[i]: np.mean([float(r[fns[i]]) for r in rows]) for i in range(1, len(fns))
        })

    with open(os.path.join(out_dir, 'evaluations.csv'), 'a') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writerow(avg)


def train_lsvc(x, y, usr_num, fns, lsvcs_dir):
    c = LinearSVC()
    c.fit(x, y)
    c.save(path=os.path.join(lsvcs_dir, mdl_save_temp.format(usr_num=usr_num)))
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


def evaluate_model(usr_num, enc_gen, enc_frg, fns, lsvcs_dir):
    x, y = get_lsvc_train_data(enc_gen, enc_frg)
    return train_lsvc(x, y, usr_num, fns, lsvcs_dir)


def get_encoded_data(e, gen_x, frg_x, msk_val):
    enc_gen = e.predict(inp=sequence.pad_sequences(gen_x, value=msk_val))  # Encoded Genuine Data
    enc_frg = e.predict(inp=sequence.pad_sequences(frg_x, value=msk_val))  # Encoded Forged Data
    return enc_gen, enc_frg


def get_autoencoder_train_data(data, usr_num, msk_val):
    (gen_x, gen_y) = data.get_genuine_combinations(usr_num)
    x, y = sequence.pad_sequences(gen_x, value=msk_val), sequence.pad_sequences(gen_y, value=msk_val)
    return x, y, sequence.pad_sequences(data.gen[usr_num], value=msk_val), \
        sequence.pad_sequences(data.frg[usr_num], value=msk_val)


def prepare_evaluations_csv(out_dir, fns):
    with open(os.path.join(out_dir, 'evaluations.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()


def prepare_output_directories(epc, earc, darc, ct, bd):
    out_dir = output_directory_temp.format(
        bd='b' if bd else '', ct=ct, earc='x'.join(map(str, earc)), darc='x'.join(map(str, darc)), epc=epc
    )
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    aes_dir = os.path.join(out_dir, 'autoencoders')
    if not os.path.exists(aes_dir):
        os.mkdir(aes_dir)

    lsvcs_dir = os.path.join(out_dir, 'linear_svcs')
    if not os.path.exists(lsvcs_dir):
        os.mkdir(lsvcs_dir)

    return out_dir, aes_dir, lsvcs_dir


def process_models(data, epc, earc, darc, ct, bd, msk_val, fns):
    out_dir, aes_dir, lsvcs_dir = prepare_output_directories(epc, earc, darc, ct, bd)
    prepare_evaluations_csv(out_dir, fns)
    for usr_num in range(usr_cnt):
        x, y, gen_x, frg_x = get_autoencoder_train_data(data, usr_num, msk_val)
        e = load_encoder(x, y, epc, earc, darc, ct, bd, usr_num + 1, msk_val, aes_dir)
        enc_gen, enc_frg = get_encoded_data(e, gen_x, frg_x, msk_val)
        evl = evaluate_model(usr_num + 1, enc_gen, enc_frg, fns, lsvcs_dir)
        evl.update({
            fns[1]: np.mean(list(map(len, data.gen[usr_num]))),
            fns[2]: np.mean(list(map(len, data.frg[usr_num])))
        })
        save_evaluation(evl, fns, out_dir)
    save_avg_evaluation(fns, out_dir)

if __name__ == '__main__':
    process_models(get_data(), ae_tr_epochs, enc_arc, dec_arc, cell_type, bd_cell_type, mask_value, csv_fns)
