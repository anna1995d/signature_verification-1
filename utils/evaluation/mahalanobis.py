import csv
import os

import numpy as np
from scipy import spatial

from utils.config import CONFIG


def prepare_mahalanobis_output_directory(out_dir):
    mdst_dir = os.path.join(out_dir, 'mahalanobis')
    if not os.path.exists(mdst_dir):
        os.mkdir(mdst_dir)

    for usr_num in range(1, CONFIG.usr_cnt + 1):
        mdst_usr_dir = os.path.join(mdst_dir, 'U{usr_num}'.format(usr_num=usr_num))
        if not os.path.exists(mdst_usr_dir):
            os.mkdir(mdst_usr_dir)

    return mdst_dir


def get_mahalanobis_distances(tr_enc_gen, enc_gen, enc_frg):
    cov_diag = np.diag(np.cov(tr_enc_gen, rowvar=False)).copy()
    cov_diag[np.where(cov_diag == 0.0)] += np.finfo(np.float64).eps
    cov_inv = np.linalg.inv(np.diag(cov_diag))
    mean = np.mean(tr_enc_gen, axis=0)
    return sorted([spatial.distance.mahalanobis(enc, mean, cov_inv) for enc in enc_gen]), \
        sorted([spatial.distance.mahalanobis(enc, mean, cov_inv) for enc in enc_frg])


def evaluate_mahalanobis(usr_num, tr_enc_gen, enc_gen, enc_frg):
    enc_gen_mdst, enc_frg_mdst = get_mahalanobis_distances(tr_enc_gen, enc_gen, enc_frg)
    enc_mdst = [[mdst, 0] for mdst in enc_gen_mdst] + [[mdst, 0] for mdst in enc_frg_mdst]
    for dst in enc_mdst:
        dst[1] += np.where(enc_gen_mdst <= dst[0])[0].shape[0] + np.where(enc_frg_mdst > dst[0])[0].shape[0]
    trs, prc = max(enc_mdst, key=lambda x: x[1])
    return {
        'Writer No': usr_num,
        'Precision': prc / len(enc_mdst),
        'Threshold': trs
    }


def prepare_mahalanobis_distances_evaluations_csv(mdst_dir):
    with open(os.path.join(mdst_dir, 'evaluations.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.mhln_csv_fns)
        w.writeheader()


def save_mahalanobis_evaluation(usr_num, mdst_dir, enc_gen_mdst, enc_frg_mdst, evl):
    with open(os.path.join(mdst_dir, 'U{usr_num}/mahalanobis_distances_genuine.dat'.format(usr_num=usr_num)), 'w') as f:
        for dst in enc_gen_mdst:
            f.write('{dst}\n'.format(dst=dst))

    with open(os.path.join(mdst_dir, 'U{usr_num}/mahalanobis_distances_forged.dat'.format(usr_num=usr_num)), 'w') as f:
        for dst in enc_frg_mdst:
            f.write('{dst}\n'.format(dst=dst))

    with open(os.path.join(mdst_dir, 'evaluations.csv'), 'a') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.mhln_csv_fns)
        w.writerow(evl)


def save_mahalanobis_avg_evaluation(mdst_dir):
    with open(os.path.join(mdst_dir, 'evaluations.csv'), 'r') as f:
        avg = {
            CONFIG.mhln_csv_fns[0]: 'AVG'
        }
        rows = [r for r in csv.DictReader(f, fieldnames=CONFIG.mhln_csv_fns)][1:]
        avg.update({
            CONFIG.mhln_csv_fns[i]: np.mean(
                [float(r[CONFIG.mhln_csv_fns[i]]) for r in rows]
            ) for i in range(1, len(CONFIG.mhln_csv_fns))
        })

    with open(os.path.join(mdst_dir, 'evaluations.csv'), 'a') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.mhln_csv_fns)
        w.writerow(avg)
