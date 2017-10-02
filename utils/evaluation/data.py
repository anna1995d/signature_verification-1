import itertools
import os

import numpy as np
from scipy.special import comb

from utils import compute_distances
from utils.config import CONFIG
from utils.data import DATA
from utils.rnn import get_encoded_data


def _get_evaluation_data(encoder, writer_iterator):
    encoded_distances = dict()
    x, y = list(), list()
    for writer in writer_iterator:
        reference_encoded_genuine, encoded_genuine, encoded_forgery = [
            get_encoded_data(encoder, DATA.gen_x[writer][:CONFIG.ref_smp_cnt]),
            get_encoded_data(encoder, DATA.gen_x[writer][CONFIG.ref_smp_cnt:]),
            get_encoded_data(encoder, DATA.frg_x[writer])
        ]

        encoded_distances.update({
            'G{wrt}'.format(wrt=writer): compute_distances(encoded_genuine),
            'F{wrt}'.format(wrt=writer): compute_distances(encoded_genuine, encoded_forgery)
        })

        reference_distances, genuine_distances, forgery_distances = [
            compute_distances(reference_encoded_genuine),
            compute_distances(encoded_genuine, reference_encoded_genuine),
            compute_distances(encoded_forgery, reference_encoded_genuine),
        ]

        reference_mean_distances = np.mean(reference_distances, axis=1)
        feat_vec = np.array([
            np.mean(np.min(reference_distances, axis=1)),
            np.min(reference_mean_distances),
            np.mean(np.max(reference_distances, axis=1))
        ], ndmin=2)

        genuine_x = np.nan_to_num((np.concatenate([
            np.min(genuine_distances, axis=1, keepdims=True),
            np.mean(genuine_distances[:, np.argmin(reference_mean_distances)].reshape((-1, 1)), axis=1, keepdims=True),
            np.max(genuine_distances, axis=1, keepdims=True)
        ], axis=1) / feat_vec))
        forgery_x = np.nan_to_num((np.concatenate([
            np.min(forgery_distances, axis=1, keepdims=True),
            np.mean(forgery_distances[:, np.argmin(reference_mean_distances)].reshape((-1, 1)), axis=1, keepdims=True),
            np.max(forgery_distances, axis=1, keepdims=True)
        ], axis=1) / feat_vec))
        x.append(np.concatenate([genuine_x, forgery_x]))

        genuine_y = np.ones_like(genuine_x[:, 0])
        forgery_y = np.zeros_like(forgery_x[:, 0])
        y.append(np.concatenate([genuine_y, forgery_y]))

    np.savez_compressed(os.path.join(CONFIG.out_dir, 'encoded_distances'), **encoded_distances)

    return np.concatenate(x), np.concatenate(y)


def get_evaluation_train_data(encoder):
    return _get_evaluation_data(encoder, range(CONFIG.clf_tr_wrt_cnt))


def get_evaluation_test_data(encoder):
    return _get_evaluation_data(encoder, range(CONFIG.clf_tr_wrt_cnt, CONFIG.clf_tr_wrt_cnt + CONFIG.clf_ts_wrt_cnt))


def get_siamese_evaluation_train_data(encoder):
    x, y = list(), list()
    for writer in range(CONFIG.clf_tr_wrt_cnt):
        encoded_genuine, encoded_forgery = [
            get_encoded_data(encoder, DATA.gen_x[writer]),
            get_encoded_data(encoder, DATA.frg_x[writer])
        ]

        x.extend(map(lambda z: np.array(z, ndmin=3), itertools.combinations(encoded_genuine, 2)))
        y.extend(np.ones((comb(len(encoded_genuine), 2, True), 1)))

        x.extend(map(lambda z: np.array(z, ndmin=3), itertools.combinations(encoded_forgery, 2)))
        y.extend(np.ones((comb(len(encoded_forgery), 2, True), 1)))

        x.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(encoded_genuine, encoded_forgery)))
        y.extend(np.zeros((len(encoded_genuine) * len(encoded_forgery), 1)))

    return list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x), 0, 1), 2))), np.concatenate(y)


def get_siamese_evaluation_test_data(encoder):
    x, y = list(), list()
    for writer in range(CONFIG.clf_tr_wrt_cnt, CONFIG.clf_tr_wrt_cnt + CONFIG.clf_ts_wrt_cnt):
        reference, encoded_genuine, encoded_forgery = [
            get_encoded_data(encoder, DATA.gen_x[writer][:CONFIG.sms_ts_ref_cnt]),
            get_encoded_data(
                encoder, DATA.gen_x[writer][CONFIG.sms_ts_ref_cnt:CONFIG.sms_ts_ref_cnt + CONFIG.sms_ts_evl_cnt]
            ),
            get_encoded_data(encoder, DATA.frg_x[writer][:CONFIG.sms_ts_evl_cnt])
        ]

        x.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(reference, encoded_genuine)))
        y.extend(np.ones((len(encoded_genuine), 1)))

        x.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(reference, encoded_forgery)))
        y.extend(np.zeros((len(encoded_forgery), 1)))

    return list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x), 0, 1), 2))), np.concatenate(y)
