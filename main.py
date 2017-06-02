#!/usr/bin/env python

from utils.config import CONFIG
from utils.evaluation.lsvc import evaluate_lsvc, prepare_lsvc_evaluations_csv, get_lsvc_train_data, train_lsvc, \
    save_lsvc_evaluation, save_lsvc_avg_evaluation
from utils.io import prepare_output_directory
from utils.rnn import get_autoencoder_train_data, get_autoencoder_evaluation_data, load_encoder, get_encoded_data


def process_models():
    prepare_output_directory()

    prepare_lsvc_evaluations_csv()

    x, y = get_autoencoder_train_data()
    e = load_encoder(x, y)

    for usr_num in range(1, CONFIG.usr_cnt + 1):
        tr_gen_x, tr_frg_x, gen_x, frg_x = get_autoencoder_evaluation_data(usr_num - 1)

        enc_gen, enc_frg = get_encoded_data(e, gen_x), get_encoded_data(e, frg_x)

        x, y = get_lsvc_train_data(enc_gen, enc_frg)
        c = train_lsvc(x, y)
        evl = evaluate_lsvc(c, x, y, usr_num)
        save_lsvc_evaluation(evl)
    save_lsvc_avg_evaluation()


if __name__ == '__main__':
    process_models()
