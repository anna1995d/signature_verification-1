import csv
import itertools
import os

import numpy as np
from sklearn.metrics import classification_report

from seq2seq.models import SiameseClassifier
from utils.config import CONFIG
from utils.data import DATA
from utils.rnn import get_encoded_data


def get_siamese_data(encoder, fold):
    x, y, x_cv, y_cv, x_ts, y_ts = list(), list(), list(), list(), list(), list()
    for writer in range(CONFIG.wrt_cnt):
        encoded_genuine, encoded_forgery = [
            get_encoded_data(encoder, DATA.gen_x[writer]),
            get_encoded_data(encoder, DATA.frg_x[writer])
        ]

        if (fold < 0 and writer >= CONFIG.tr_wrt_cnt) or (0 <= fold == writer // (CONFIG.wrt_cnt // CONFIG.spt_cnt)):
            mean = np.mean(encoded_genuine[:CONFIG.ref_smp_cnt], axis=0)
            encoded_genuine, encoded_forgery = encoded_genuine - mean, encoded_forgery - mean

            x.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_genuine[:CONFIG.ref_smp_cnt], encoded_genuine[:CONFIG.ref_smp_cnt]
            )))
            y.extend(np.ones((CONFIG.ref_smp_cnt * CONFIG.ref_smp_cnt, 1)))

            x_cv.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_genuine[:CONFIG.ref_smp_cnt], encoded_genuine[CONFIG.ref_smp_cnt:]
            )))
            y_cv.extend(np.ones(
                (len(encoded_genuine[:CONFIG.ref_smp_cnt]) * len(encoded_genuine[CONFIG.ref_smp_cnt:]), 1)
            ))

            x_cv.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_genuine[CONFIG.ref_smp_cnt:], encoded_genuine[:CONFIG.ref_smp_cnt]
            )))
            y_cv.extend(np.ones(
                (len(encoded_genuine[:CONFIG.ref_smp_cnt]) * len(encoded_genuine[CONFIG.ref_smp_cnt:]), 1)
            ))

            x_cv.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(encoded_forgery, encoded_forgery)))
            y_cv.extend(np.ones((len(encoded_forgery) * len(encoded_forgery), 1)))

            x_cv.extend(map(
                lambda z: np.array(z, ndmin=3), itertools.product(encoded_genuine[:CONFIG.ref_smp_cnt], encoded_forgery)
            ))
            y_cv.extend(np.zeros((len(encoded_genuine[:CONFIG.ref_smp_cnt]) * len(encoded_forgery), 1)))

            x_cv.extend(map(
                lambda z: np.array(z, ndmin=3), itertools.product(encoded_forgery, encoded_genuine[:CONFIG.ref_smp_cnt])
            ))
            y_cv.extend(np.zeros((len(encoded_forgery) * len(encoded_genuine[:CONFIG.ref_smp_cnt]), 1)))

            x_ts.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_genuine[:CONFIG.ref_smp_cnt], encoded_genuine[CONFIG.ref_smp_cnt:]
            )))
            y_ts.extend(np.ones((len(encoded_genuine[CONFIG.ref_smp_cnt:]), 1)))

            x_ts.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_genuine[:CONFIG.ref_smp_cnt], encoded_forgery
            )))
            y_ts.extend(np.zeros((len(encoded_forgery), 1)))
        else:
            mean = np.mean(encoded_genuine, axis=0)
            encoded_genuine, encoded_forgery = encoded_genuine - mean, encoded_forgery - mean

            x.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(encoded_genuine, encoded_genuine)))
            y.extend(np.ones((len(encoded_genuine) * len(encoded_genuine), 1)))

            x.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(encoded_forgery, encoded_forgery)))
            y.extend(np.ones((len(encoded_forgery) * len(encoded_forgery), 1)))

            x.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(encoded_genuine, encoded_forgery)))
            y.extend(np.zeros((len(encoded_genuine) * len(encoded_forgery), 1)))

            x.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(encoded_forgery, encoded_genuine)))
            y.extend(np.zeros((len(encoded_forgery) * len(encoded_genuine), 1)))

    x = list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x), 0, 1), 2)))
    y = np.concatenate(y)
    x_cv = list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x_cv), 0, 1), 2)))
    y_cv = np.concatenate(y_cv)
    x_ts = list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x_ts), 0, 1), 2)))
    y_ts = np.concatenate(y_ts)

    return x, y, x_cv, y_cv, x_ts, y_ts


def get_evaluation(x, y, x_cv, y_cv, x_ts, y_ts, fold):
    sms = SiameseClassifier(fold)
    if CONFIG.sms_md == 'train':
        sms.fit(x, y, x_cv, y_cv)
        sms.save(os.path.join(CONFIG.out_dir, 'siamese_fold{}.hdf5').format(fold))
    else:
        sms.load(os.path.join(CONFIG.out_dir, 'siamese_fold{}.hdf5').format(fold))

    y_prb = (np.reshape(sms.predict(x_ts), (-1, CONFIG.ref_smp_cnt)) >= CONFIG.sms_ts_prb_thr).astype(np.int32)
    y_prd = (np.count_nonzero(y_prb, axis=1) >= CONFIG.sms_ts_acc_thr).astype(np.int32)
    report = classification_report(y_true=y_ts, y_pred=y_prd, digits=CONFIG.clf_rpt_dgt)
    scores = list(map(float, report.split('\n')[-2].split()[3:6]))

    print(report)

    return dict(zip(CONFIG.evaluation, scores))


def save_evaluation(evaluation):
    with open(os.path.join(CONFIG.out_dir, 'evaluation.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.evaluation)
        w.writeheader()
        w.writerows(evaluation)
