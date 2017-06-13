import csv
import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import NuSVC

from utils import compute_distances
from utils.config import CONFIG
from utils.data import DATA
from utils.rnn import get_encoded_data


def _get_svc_data(e, usr_num_gen):
    x, y = list(), list()
    for usr_num in usr_num_gen:
        ref_enc_gen, enc_gen, enc_frg = [
            get_encoded_data(e, DATA.gen[usr_num][:CONFIG.svc_smp_cnt]),
            get_encoded_data(e, DATA.gen[usr_num][CONFIG.svc_smp_cnt:]),
            get_encoded_data(e, DATA.frg[usr_num])
        ]

        ref_dists, gen_dists, frg_dists = [
            compute_distances(ref_enc_gen),
            compute_distances(enc_gen, ref_enc_gen),
            compute_distances(enc_frg, ref_enc_gen),
        ]

        ref_mdists = np.mean(ref_dists, axis=1)
        feat_vec = np.array([
            np.mean(np.min(ref_dists, axis=1)), np.min(ref_mdists), np.mean(np.max(ref_dists, axis=1))
        ], ndmin=2)

        gen_x = np.nan_to_num(np.concatenate([
            np.min(gen_dists, axis=1, keepdims=True),
            np.mean(gen_dists[:, np.argmin(ref_mdists)].reshape((-1, 1)), axis=1, keepdims=True),
            np.max(gen_dists, axis=1, keepdims=True)
        ], axis=1) / feat_vec)
        frg_x = np.nan_to_num(np.concatenate([
            np.min(frg_dists, axis=1, keepdims=True),
            np.mean(frg_dists[:, np.argmin(ref_mdists)].reshape((-1, 1)), axis=1, keepdims=True),
            np.max(frg_dists, axis=1, keepdims=True)
        ], axis=1) / feat_vec)
        x.append(np.concatenate([gen_x, frg_x]))

        gen_y = np.ones_like(gen_x[:, 0])
        frg_y = np.zeros_like(frg_x[:, 0])
        y.append(np.concatenate([gen_y, frg_y]))

    return x, y


def get_svc_train_data(e):
    x, y = list(map(lambda vec: np.concatenate(vec, axis=0), _get_svc_data(e, range(CONFIG.svc_tr_usr_cnt))))
    return x, y


def train_svc(x, y):
    c = NuSVC(nu=0.7, gamma=0.01, verbose=CONFIG.verbose)
    c.fit(x, y)
    return c


def get_svc_evaluation_data(e):
    return _get_svc_data(e, range(CONFIG.svc_tr_usr_cnt, CONFIG.svc_tr_usr_cnt + CONFIG.svc_ts_usr_cnt))


def evaluate_svc(c, x, y, usr_num):
    cr = list(map(float, classification_report(y_true=y, y_pred=c.predict(x)).split('\n')[-2].split()[3:6]))
    return {
        CONFIG.svc_csv_fns[0]: usr_num,
        CONFIG.svc_csv_fns[1]: np.mean(list(map(len, DATA.gen[usr_num - 1]))),
        CONFIG.svc_csv_fns[2]: np.mean(list(map(len, DATA.frg[usr_num - 1]))),
        CONFIG.svc_csv_fns[3]: cr[0],
        CONFIG.svc_csv_fns[4]: cr[1],
        CONFIG.svc_csv_fns[5]: cr[2]
    }


def prepare_svc_evaluations_csv():
    with open(os.path.join(CONFIG.out_dir, 'svc_evaluations.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.svc_csv_fns)
        w.writeheader()


def save_svc_evaluation(evl):
    with open(os.path.join(CONFIG.out_dir, 'svc_evaluations.csv'), 'a') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.svc_csv_fns)
        w.writerow(evl)


def save_svc_avg_evaluation():
    with open(os.path.join(CONFIG.out_dir, 'svc_evaluations.csv'), 'r') as f:
        avg = {
            CONFIG.svc_csv_fns[0]: 'AVG'
        }
        rows = [r for r in csv.DictReader(f, fieldnames=CONFIG.svc_csv_fns)][1:]
        avg.update({
            CONFIG.svc_csv_fns[i]: np.mean(
                [float(r[CONFIG.svc_csv_fns[i]]) for r in rows]
            ) for i in range(1, len(CONFIG.svc_csv_fns))
        })
    save_svc_evaluation(avg)
