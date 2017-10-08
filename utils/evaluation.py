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


def get_siamese_evaluation_train_data(encoder, fold):
    x, y, x_cv, y_cv = list(), list(), list(), list()
    for writer in range(CONFIG.wrt_cnt):
        encoded_genuine, encoded_forgery = [
            get_encoded_data(encoder, DATA.gen_x[writer]),
            get_encoded_data(encoder, DATA.frg_x[writer])
        ]

        genuine_genuine_x = map(lambda z: np.array(z, ndmin=3), itertools.combinations(encoded_genuine, 2))
        genuine_genuine_y = np.ones((comb(len(encoded_genuine), 2, True), 1))

        forgery_forgery_x = map(lambda z: np.array(z, ndmin=3), itertools.combinations(encoded_forgery, 2))
        forgery_forgery_y = np.ones((comb(len(encoded_forgery), 2, True), 1))

        genuine_forgery_x = map(lambda z: np.array(z, ndmin=3), itertools.product(encoded_genuine, encoded_forgery))
        genuine_forgery_y = np.zeros((len(encoded_genuine) * len(encoded_forgery), 1))

        if 0 <= fold == writer // (CONFIG.wrt_cnt // CONFIG.spt_cnt) or (fold < 0 and writer >= CONFIG.tr_wrt_cnt):
            x_cv.extend(genuine_genuine_x)
            y_cv.extend(genuine_genuine_y)

            x_cv.extend(forgery_forgery_x)
            y_cv.extend(forgery_forgery_y)

            x_cv.extend(genuine_forgery_x)
            y_cv.extend(genuine_forgery_y)
        else:
            x.extend(genuine_genuine_x)
            y.extend(genuine_genuine_y)

            x.extend(forgery_forgery_x)
            y.extend(forgery_forgery_y)

            x.extend(genuine_forgery_x)
            y.extend(genuine_forgery_y)

    return list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x), 0, 1), 2))), np.concatenate(y), \
        list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x_cv), 0, 1), 2))), np.concatenate(y_cv)


def get_siamese_evaluation_test_data(encoder, fold):
    x, y = list(), list()
    for writer in range(CONFIG.wrt_cnt):
        if 0 <= fold != writer // (CONFIG.wrt_cnt // CONFIG.spt_cnt) or (fold < 0 and writer < CONFIG.tr_wrt_cnt):
            continue

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


def get_optimized_evaluation(x_train, y_train, x_cv, y_cv, x_test, y_test, fold):
    sms = SiameseClassifier(fold)
    if CONFIG.sms_md == 'train':
        sms.fit(x_train, y_train, x_cv, y_cv)
        sms.save(os.path.join(CONFIG.out_dir, 'siamese_fold{}.hdf5').format(fold))
    else:
        sms.load(os.path.join(CONFIG.out_dir, 'siamese_fold{}.hdf5').format(fold))

    y_pred = (np.mean(np.reshape(sms.predict(x_test), (-1, CONFIG.sms_ts_ref_cnt)), axis=1) >= 0.5).astype(np.int32)
    scores = list(map(float, classification_report(
        y_true=y_test, y_pred=y_pred, digits=CONFIG.clf_rpt_dgt
    ).split('\n')[-2].split()[3:6]))

    return dict(zip(CONFIG.evaluation, scores))


def save_evaluation(evaluation):
    with open(os.path.join(CONFIG.out_dir, 'evaluation.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.evaluation)
        w.writeheader()
        w.writerows(evaluation)
