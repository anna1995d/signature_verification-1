import csv
import os

import numpy as np
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import NuSVC

from utils import compute_distances
from utils.config import CONFIG
from utils.data import DATA
from utils.rnn import get_encoded_data


def _scorer(y, y_pred):
    scores = list(map(float, classification_report(y_true=y, y_pred=y_pred).split('\n')[-2].split()[3:6]))
    return scores[0]


def _get_svc_data(e, usr_num_gen):
    x, y = list(), list()
    for usr_num in usr_num_gen:
        ref_enc_gen, enc_gen, enc_frg = [
            get_encoded_data(e, DATA.gen_x[usr_num][:CONFIG.svc_smp_cnt]),
            get_encoded_data(e, DATA.gen_x[usr_num][CONFIG.svc_smp_cnt:]),
            get_encoded_data(e, DATA.frg_x[usr_num])
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

        gen_x = np.nan_to_num((np.concatenate([
            np.min(gen_dists, axis=1, keepdims=True),
            np.mean(gen_dists[:, np.argmin(ref_mdists)].reshape((-1, 1)), axis=1, keepdims=True),
            np.max(gen_dists, axis=1, keepdims=True)
        ], axis=1) - feat_vec) * 100)
        frg_x = np.nan_to_num((np.concatenate([
            np.min(frg_dists, axis=1, keepdims=True),
            np.mean(frg_dists[:, np.argmin(ref_mdists)].reshape((-1, 1)), axis=1, keepdims=True),
            np.max(frg_dists, axis=1, keepdims=True)
        ], axis=1) - feat_vec) * 100)
        x.append(np.concatenate([gen_x, frg_x]))

        gen_y = np.ones_like(gen_x[:, 0])
        frg_y = np.zeros_like(frg_x[:, 0])
        y.append(np.concatenate([gen_y, frg_y]))

    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)


def get_svc_train_data(e):
    return _get_svc_data(e, range(CONFIG.svc_tr_usr_cnt))


def get_optimized_svc_evaluation(x_train, y_train, x_cv, y_cv):
    x, y = np.concatenate([x_train, x_cv]), np.concatenate([y_train, y_cv])
    estimator = NuSVC()
    param_grid = [{
        'kernel': ['rbf', 'sigmoid'],
        'nu': np.arange(start=0.600, stop=0.850, step=0.001, dtype=np.float64),
        'gamma': [
            0.1, 0.2, 0.3,
            0.01, 0.02, 0.03,
            0.001, 0.002, 0.003,
            0.0001, 0.0002, 0.0003,
            0.00001, 0.00002, 0.00003,
            0.000001, 0.000002, 0.000003,
            0.0000001, 0.0000002, 0.0000003,
            1e-7, 2e-7, 3e-7, 1e-8, 2e-8, 3e-8,
            1e-9, 2e-9, 3e-9, 1e-10, 2e-10, 3e-10
        ]
    }]
    scoring = make_scorer(_scorer)
    cv = PredefinedSplit(test_fold=np.concatenate([np.ones_like(x_train[:, 0]) * (-1), np.zeros_like(x_cv[:, 0])]))
    c = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, return_train_score=False)
    c.fit(x, y)

    return {
        CONFIG.svc_csv_fns[0]: c.best_params_['kernel'],
        CONFIG.svc_csv_fns[1]: c.best_params_['nu'],
        CONFIG.svc_csv_fns[2]: c.best_params_['gamma'],
        CONFIG.svc_csv_fns[3]: c.best_score_
    }


def get_svc_evaluation_data(e):
    return _get_svc_data(e, range(CONFIG.svc_tr_usr_cnt, CONFIG.svc_tr_usr_cnt + CONFIG.svc_ts_usr_cnt))


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
