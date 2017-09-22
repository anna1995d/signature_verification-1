import os

import numpy as np

from utils import compute_distances
from utils.config import CONFIG
from utils.data import DATA
from utils.rnn import get_encoded_data


def _get_evaluation_data(e, usr_iter):
    dists = dict()
    x, y = list(), list()
    for usr in usr_iter:
        ref_enc_gen, enc_gen, enc_frg = [
            get_encoded_data(e, DATA.gen_x[usr][:CONFIG.ref_smp_cnt]),
            get_encoded_data(e, DATA.gen_x[usr][CONFIG.ref_smp_cnt:]),
            get_encoded_data(e, DATA.frg_x[usr])
        ]

        dists.update({
            'gen_{usr}'.format(usr=usr): compute_distances(enc_gen),
            'frg_{usr}'.format(usr=usr): compute_distances(enc_gen, enc_frg)
        })

        ref_dists, gen_dists, frg_dists = [
            compute_distances(ref_enc_gen),
            compute_distances(enc_gen, ref_enc_gen),
            compute_distances(enc_frg, ref_enc_gen),
        ]

        ref_mdists = np.mean(ref_dists, axis=1)
        feat_vec = np.array([
            np.mean(np.min(ref_dists, axis=1)), np.min(ref_mdists), np.mean(np.max(ref_dists, axis=1))
        ], ndmin=2)

        gen_x = np.nan_to_num((np.concatenate([
            np.min(gen_dists, axis=1, keepdims=True),
            np.mean(gen_dists[:, np.argmin(ref_mdists)].reshape((-1, 1)), axis=1, keepdims=True),
            np.max(gen_dists, axis=1, keepdims=True)
        ], axis=1) / feat_vec))
        frg_x = np.nan_to_num((np.concatenate([
            np.min(frg_dists, axis=1, keepdims=True),
            np.mean(frg_dists[:, np.argmin(ref_mdists)].reshape((-1, 1)), axis=1, keepdims=True),
            np.max(frg_dists, axis=1, keepdims=True)
        ], axis=1) / feat_vec))
        x.append(np.concatenate([gen_x, frg_x]))

        gen_y = np.ones_like(gen_x[:, 0])
        frg_y = np.zeros_like(frg_x[:, 0])
        y.append(np.concatenate([gen_y, frg_y]))

    np.savez_compressed(os.path.join(CONFIG.out_dir, 'distances'), **dists)

    return np.concatenate(x), np.concatenate(y)


def get_evaluation_train_data(e):
    return _get_evaluation_data(e, range(CONFIG.clf_tr_usr_cnt))


def get_evaluation_test_data(e):
    start = CONFIG.clf_tr_usr_cnt
    return _get_evaluation_data(e, range(start, start + CONFIG.clf_ts_usr_cnt))
