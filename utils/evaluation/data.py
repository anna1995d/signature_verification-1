import os

import numpy as np

from utils import compute_distances
from utils.config import CONFIG
from utils.data import DATA
from utils.rnn import get_encoded_data


def _get_evaluation_data(encoder, writer_iterator):
    dists = dict()
    x, y = list(), list()
    for writer in writer_iterator:
        ref_enc_gen, enc_gen, enc_frg = [
            get_encoded_data(encoder, DATA.gen_x[writer][:CONFIG.ref_smp_cnt]),
            get_encoded_data(encoder, DATA.gen_x[writer][CONFIG.ref_smp_cnt:]),
            get_encoded_data(encoder, DATA.frg_x[writer])
        ]

        dists.update({
            'gen_{wrt}'.format(wrt=writer): compute_distances(enc_gen),
            'frg_{wrt}'.format(wrt=writer): compute_distances(enc_gen, enc_frg)
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


def get_evaluation_train_data(encoder):
    return _get_evaluation_data(encoder, range(CONFIG.clf_tr_wrt_cnt))


def get_evaluation_test_data(encoder):
    start = CONFIG.clf_tr_wrt_cnt
    return _get_evaluation_data(encoder, range(start, start + CONFIG.clf_ts_wrt_cnt))
