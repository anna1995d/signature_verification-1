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
    x, y, x_cv, y_cv, x_ts_1, y_ts_1, x_ts_2, y_ts_2 = list(), list(), list(), list(), list(), list(), list(), list()
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
                encoded_genuine[CONFIG.ref_smp_cnt:], encoded_genuine[CONFIG.ref_smp_cnt:]
            )))
            y_cv.extend(np.ones(
                (len(encoded_genuine[CONFIG.ref_smp_cnt:]) * len(encoded_genuine[CONFIG.ref_smp_cnt:]), 1)
            ))

            x_cv.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(encoded_forgery, encoded_forgery)))
            y_cv.extend(np.ones((len(encoded_forgery) * len(encoded_forgery), 1)))

            x_cv.extend(map(
                lambda z: np.array(z, ndmin=3), itertools.product(encoded_genuine[CONFIG.ref_smp_cnt:], encoded_forgery)
            ))
            y_cv.extend(np.zeros((len(encoded_genuine[CONFIG.ref_smp_cnt:]) * len(encoded_forgery), 1)))

            x_cv.extend(map(
                lambda z: np.array(z, ndmin=3), itertools.product(encoded_forgery, encoded_genuine[CONFIG.ref_smp_cnt:])
            ))
            y_cv.extend(np.zeros((len(encoded_forgery) * len(encoded_genuine[CONFIG.ref_smp_cnt:]), 1)))

            x_ts_1.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_genuine[:CONFIG.ref_smp_cnt], encoded_genuine[CONFIG.ref_smp_cnt:]
            )))
            y_ts_1.extend(np.ones((len(encoded_genuine[CONFIG.ref_smp_cnt:]), 1)))

            x_ts_1.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_genuine[:CONFIG.ref_smp_cnt], encoded_forgery
            )))
            y_ts_1.extend(np.zeros((len(encoded_forgery), 1)))

            x_ts_2.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_genuine[CONFIG.ref_smp_cnt:], encoded_genuine[:CONFIG.ref_smp_cnt]
            )))
            y_ts_2.extend(np.ones((len(encoded_genuine[CONFIG.ref_smp_cnt:]), 1)))

            x_ts_2.extend(map(lambda z: np.array(z, ndmin=3), itertools.product(
                encoded_forgery, encoded_genuine[:CONFIG.ref_smp_cnt]
            )))
            y_ts_2.extend(np.zeros((len(encoded_forgery), 1)))
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
    x_ts_1 = list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x_ts_1), 0, 1), 2)))
    y_ts_1 = np.concatenate(y_ts_1)
    x_ts_2 = list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x_ts_2), 0, 1), 2)))
    y_ts_2 = np.concatenate(y_ts_2)

    return x, y, x_cv, y_cv, x_ts_1, y_ts_1, x_ts_2, y_ts_2


def get_evaluation(x, y, x_cv, y_cv, x_ts_1, y_ts_1, x_ts_2, y_ts_2, fold):
    sms = SiameseClassifier(fold)
    if CONFIG.sms_md == 'train':
        sms.fit(x, y, x_cv, y_cv)
        sms.save(os.path.join(CONFIG.out_dir, 'siamese_fold{}.hdf5').format(fold))
    else:
        sms.load(os.path.join(CONFIG.out_dir, 'siamese_fold{}.hdf5').format(fold))

    y_prb_1 = (np.reshape(sms.predict(x_ts_1), (-1, CONFIG.ref_smp_cnt)) >= CONFIG.sms_ts_prb_thr).astype(np.int32)
    y_prd_1 = (np.count_nonzero(y_prb_1, axis=1) >= CONFIG.sms_ts_acc_thr).astype(np.int32)
    report_1 = classification_report(y_true=y_ts_1, y_pred=y_prd_1, digits=CONFIG.clf_rpt_dgt)
    scores_1 = list(map(float, report_1.split('\n')[-2].split()[3:6]))

    print(report_1)

    y_prb_2 = (np.reshape(sms.predict(x_ts_2), (-1, CONFIG.ref_smp_cnt)) >= CONFIG.sms_ts_prb_thr).astype(np.int32)
    y_prd_2 = (np.count_nonzero(y_prb_2, axis=1) >= CONFIG.sms_ts_acc_thr).astype(np.int32)
    report_2 = classification_report(y_true=y_ts_2, y_pred=y_prd_2, digits=CONFIG.clf_rpt_dgt)

    print(report_2)

    y_prb_3 = (
        np.reshape(sms.predict(x_ts_1), (-1, CONFIG.ref_smp_cnt)) +
        np.reshape(sms.predict(x_ts_2), (-1, CONFIG.ref_smp_cnt)) >= CONFIG.sms_ts_prb_thr * 2
    ).astype(np.int32)
    y_prd_3 = (np.count_nonzero(y_prb_3, axis=1) >= CONFIG.sms_ts_acc_thr).astype(np.int32)
    report_3 = classification_report(y_true=y_ts_1, y_pred=y_prd_3, digits=CONFIG.clf_rpt_dgt)

    print(report_3)

    y_prb_4 = (
        np.maximum(np.reshape(sms.predict(x_ts_1), (-1, CONFIG.ref_smp_cnt)),
                   np.reshape(sms.predict(x_ts_2), (-1, CONFIG.ref_smp_cnt))) >= CONFIG.sms_ts_prb_thr
    ).astype(np.int32)
    y_prd_4 = (np.count_nonzero(y_prb_4, axis=1) >= CONFIG.sms_ts_acc_thr).astype(np.int32)
    report_4 = classification_report(y_true=y_ts_1, y_pred=y_prd_4, digits=CONFIG.clf_rpt_dgt)

    print(report_4)

    y_prb_5 = (
        np.minimum(np.reshape(sms.predict(x_ts_1), (-1, CONFIG.ref_smp_cnt)),
                   np.reshape(sms.predict(x_ts_2), (-1, CONFIG.ref_smp_cnt))) >= CONFIG.sms_ts_prb_thr
    ).astype(np.int32)
    y_prd_5 = (np.count_nonzero(y_prb_5, axis=1) >= CONFIG.sms_ts_acc_thr).astype(np.int32)
    report_5 = classification_report(y_true=y_ts_1, y_pred=y_prd_5, digits=CONFIG.clf_rpt_dgt)

    print(report_5)

    return dict(zip(CONFIG.evaluation, scores_1))


def save_evaluation(evaluation):
    with open(os.path.join(CONFIG.out_dir, 'evaluation.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.evaluation)
        w.writeheader()
        w.writerows(evaluation)
