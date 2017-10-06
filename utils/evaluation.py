import csv
import itertools
import os

import numpy as np
from scipy.special import comb
from sklearn.metrics import classification_report

from seq2seq.models import SiameseClassifier
from utils.config import CONFIG
from utils.data import DATA
from utils.rnn import get_encoded_data


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


def get_optimized_evaluation(x_tr, y_tr, x_ts, y_ts):
    sms = SiameseClassifier()
    sms.fit(x_tr, y_tr)
    sms.save(path=os.path.join(CONFIG.out_dir, 'siamese.hdf5'))

    y_pred = (np.mean(np.reshape(sms.predict(x_ts), (-1, CONFIG.sms_ts_ref_cnt)), axis=1) >= 0.5).astype(np.int32)
    scores = list(map(
        float, classification_report(y_true=y_ts, y_pred=y_pred, digits=CONFIG.clf_rpt_dgt).split('\n')[-2].split()[3:6]
    ))

    return dict(zip(CONFIG.evaluation, scores))


def save_evaluation(evaluation):
    with open(os.path.join(CONFIG.out_dir, 'evaluation.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.evaluation)
        w.writeheader()
        w.writerow(evaluation)
