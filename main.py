#!/usr/bin/env python

from utils.config import CONFIG
from utils.data import DATA
from utils.evaluation.lsvc import prepare_lsvc_output_directory, evaluate_lsvc, prepare_lsvc_evaluations_csv, \
    save_lsvc_evaluation, save_lsvc_avg_evaluation
from utils.io import save_encoded_representations, prepare_output_directory
from utils.rnn import get_autoencoder_train_data, load_encoder, get_encoded_data


def process_models():
    prepare_output_directory()
    lsvcs_dir = prepare_lsvc_output_directory(CONFIG.out_dir)
    prepare_lsvc_evaluations_csv(lsvcs_dir)
    for usr_num in range(1, CONFIG.usr_cnt + 1):
        x, y, tr_gen_x, gen_x, frg_x = get_autoencoder_train_data(DATA, usr_num - 1)
        e = load_encoder(x, y, usr_num)

        tr_enc_gen = get_encoded_data(e, tr_gen_x)
        enc_gen = get_encoded_data(e, gen_x)
        enc_frg = get_encoded_data(e, frg_x)

        print(enc_gen)
        print(enc_frg)

        save_encoded_representations(usr_num, tr_enc_gen, enc_gen, enc_frg)

        import numpy as np
        evl = evaluate_lsvc(usr_num, np.nan_to_num(enc_gen), np.nan_to_num(enc_frg), lsvcs_dir)
        evl.update({
            CONFIG.lsvc_csv_fns[1]: np.mean(list(map(len, DATA.gen[usr_num - 1]))),
            CONFIG.lsvc_csv_fns[2]: np.mean(list(map(len, DATA.frg[usr_num - 1])))
        })
        save_lsvc_evaluation(evl, lsvcs_dir)
    save_lsvc_avg_evaluation(lsvcs_dir)


if __name__ == '__main__':
    process_models()
