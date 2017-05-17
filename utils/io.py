import os

import numpy as np

from utils.config import CONFIG


def prepare_output_directory():
    if not os.path.exists(CONFIG.out_dir):
        os.mkdir(CONFIG.out_dir)

    if not os.path.exists(CONFIG.aes_dir):
        os.mkdir(CONFIG.aes_dir)

    if not os.path.exists(CONFIG.enc_dir):
        os.mkdir(CONFIG.enc_dir)

    for usr_num in range(1, CONFIG.usr_cnt + 1):
        enc_usr_dir = os.path.join(CONFIG.enc_dir, 'U{usr_num}'.format(usr_num=usr_num))
        if not os.path.exists(enc_usr_dir):
            os.mkdir(enc_usr_dir)


def save_encoded_representations(usr_num, tr_enc_gen, enc_gen, enc_frg):
    np.save(os.path.join(CONFIG.enc_dir, 'U{usr_num}/tr_enc.csv.npy'.format(usr_num=usr_num)), tr_enc_gen)
    np.save(os.path.join(CONFIG.enc_dir, 'U{usr_num}/gen_enc.csv.npy'.format(usr_num=usr_num)), enc_gen)
    np.save(os.path.join(CONFIG.enc_dir, 'U{usr_num}/frg_enc.csv.npy'.format(usr_num=usr_num)), enc_frg)


def load_encoded_representations(usr_num):
    return np.load(os.path.join(CONFIG.enc_dir, 'U{usr_num}/tr_enc.csv.npy'.format(usr_num=usr_num))), \
           np.load(os.path.join(CONFIG.enc_dir, 'U{usr_num}/gen_enc.csv.npy'.format(usr_num=usr_num))), \
           np.load(os.path.join(CONFIG.enc_dir, 'U{usr_num}/frg_enc.csv.npy'.format(usr_num=usr_num)))
