#!/usr/bin/env python

from utils.config import CONFIG
from utils.data import DATA
from utils.io import save_encoded_representations, prepare_output_directory
from utils.rnn import get_autoencoder_train_data, load_encoder, get_encoded_data


def process_models():
    prepare_output_directory()
    for usr_num in range(1, CONFIG.usr_cnt + 1):
        x, y, tr_gen_x, gen_x, frg_x = get_autoencoder_train_data(DATA, usr_num - 1)
        e = load_encoder(x, y, usr_num)

        tr_enc_gen = get_encoded_data(e, tr_gen_x)
        enc_gen = get_encoded_data(e, gen_x)
        enc_frg = get_encoded_data(e, frg_x)

        save_encoded_representations(usr_num, tr_enc_gen, enc_gen, enc_frg)

if __name__ == '__main__':
    process_models()
