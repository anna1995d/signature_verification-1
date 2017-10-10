import csv
import itertools
import os

import numpy as np
from keras.preprocessing import sequence
from scipy.special import comb
from sklearn.metrics import classification_report

from seq2seq.models import SiameseClassifier
from utils.config import CONFIG
from utils.data import DATA


def get_siamese_evaluation_train_data(fold):
    x, y, x_cv, y_cv = list(), list(), list(), list()
    for writer in range(CONFIG.wrt_cnt):
        genuine, forgery = [
            sequence.pad_sequences(DATA.gen_x[writer], maxlen=DATA.max_len),
            sequence.pad_sequences(DATA.frg_x[writer], maxlen=DATA.max_len)
        ]

        genuine_genuine_x = map(lambda z: np.array(z, ndmin=4), itertools.combinations(genuine, 2))
        genuine_genuine_y = np.ones((comb(len(genuine), 2, True), 1))

        forgery_forgery_x = map(lambda z: np.array(z, ndmin=4), itertools.combinations(forgery, 2))
        forgery_forgery_y = np.ones((comb(len(forgery), 2, True), 1))

        genuine_forgery_x = map(lambda z: np.array(z, ndmin=4), itertools.product(genuine, forgery))
        genuine_forgery_y = np.zeros((len(genuine) * len(forgery), 1))

        if 0 <= fold == writer // (CONFIG.wrt_cnt // CONFIG.spt_cnt):
            x_cv.extend(genuine_genuine_x)
            y_cv.extend(genuine_genuine_y)

            x_cv.extend(forgery_forgery_x)
            y_cv.extend(forgery_forgery_y)

            x_cv.extend(genuine_forgery_x)
            y_cv.extend(genuine_forgery_y)
        elif fold < 0 and writer >= CONFIG.tr_wrt_cnt:
            genuine_genuine_x = map(
                lambda z: np.array(z, ndmin=4), itertools.combinations(genuine[:CONFIG.ref_smp_cnt], 2)
            )
            genuine_genuine_y = np.ones((comb(len(genuine[:CONFIG.ref_smp_cnt]), 2, True), 1))

            x.extend(genuine_genuine_x)
            y.extend(genuine_genuine_y)

            genuine_genuine_x = map(
                lambda z: np.array(z, ndmin=4), itertools.combinations(genuine[CONFIG.ref_smp_cnt:], 2)
            )
            genuine_genuine_y = np.ones((comb(len(genuine[CONFIG.ref_smp_cnt:]), 2, True), 1))

            x_cv.extend(genuine_genuine_x)
            y_cv.extend(genuine_genuine_y)

            x_cv.extend(forgery_forgery_x)
            y_cv.extend(forgery_forgery_y)

            genuine_forgery_x = map(
                lambda z: np.array(z, ndmin=4), itertools.product(genuine[CONFIG.ref_smp_cnt:], forgery)
            )
            genuine_forgery_y = np.zeros((len(genuine[CONFIG.ref_smp_cnt:]) * len(forgery), 1))

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


def get_siamese_evaluation_test_data(fold):
    x, y = list(), list()
    for writer in range(CONFIG.wrt_cnt):
        if 0 <= fold != writer // (CONFIG.wrt_cnt // CONFIG.spt_cnt) or (fold < 0 and writer < CONFIG.tr_wrt_cnt):
            continue

        reference, genuine, forgery = [
            sequence.pad_sequences(DATA.gen_x[writer][:CONFIG.ref_smp_cnt], maxlen=DATA.max_len),
            sequence.pad_sequences(DATA.gen_x[writer][CONFIG.ref_smp_cnt:], maxlen=DATA.max_len),
            sequence.pad_sequences(DATA.frg_x[writer], maxlen=DATA.max_len)
        ]

        x.extend(map(lambda z: np.array(z, ndmin=4), itertools.product(reference, genuine)))
        y.extend(np.ones((len(genuine), 1)))

        x.extend(map(lambda z: np.array(z, ndmin=4), itertools.product(reference, forgery)))
        y.extend(np.zeros((len(forgery), 1)))

    return list(map(np.squeeze, np.split(np.swapaxes(np.concatenate(x), 0, 1), 2))), np.concatenate(y)


def get_optimized_evaluation(encoder, x_train, y_train, x_cv, y_cv, x_test, y_test, fold):
    sms = SiameseClassifier(encoder, fold)
    if CONFIG.sms_md == 'train':
        sms.fit(x_train, y_train, x_cv, y_cv)
        sms.save(os.path.join(CONFIG.out_dir, 'siamese_fold{}.hdf5').format(fold))
    else:
        sms.load(os.path.join(CONFIG.out_dir, 'siamese_fold{}.hdf5').format(fold))

    y_prb = (np.reshape(sms.predict(x_test), (-1, CONFIG.ref_smp_cnt)) >= CONFIG.sms_ts_prb_thr).astype(np.int32)
    y_prd = (np.count_nonzero(y_prb, axis=1) >= CONFIG.sms_ts_acc_thr).astype(np.int32)
    report = classification_report(y_true=y_test, y_pred=y_prd, digits=CONFIG.clf_rpt_dgt)
    scores = list(map(float, report.split('\n')[-2].split()[3:6]))

    print(report)

    return dict(zip(CONFIG.evaluation, scores))


def save_evaluation(evaluation):
    with open(os.path.join(CONFIG.out_dir, 'evaluation.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.evaluation)
        w.writeheader()
        w.writerows(evaluation)
