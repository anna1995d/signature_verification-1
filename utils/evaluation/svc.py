import csv
import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import NuSVC

from utils.config import CONFIG
from utils.data import DATA
from utils.rnn import get_encoded_data


def get_svc_train_data(e):
    gen, frg, gen_mean = list(), list(), list()
    for usr_num in range(CONFIG.usr_cnt):
        enc_gen, enc_frg = get_encoded_data(e, DATA.gen[usr_num][:CONFIG.ae_smp_cnt]), \
                           get_encoded_data(e, DATA.frg[usr_num][:CONFIG.ae_smp_cnt])
        enc_gen_mean, enc_frg_mean = np.mean(enc_gen, axis=0), np.mean(enc_frg, axis=0)

        gen.append(enc_gen - enc_gen_mean)
        frg.append(enc_frg - enc_frg_mean)
        gen_mean.append(enc_gen_mean)

    enc_gen, enc_frg = np.concatenate(gen, axis=0), np.concatenate(frg, axis=0)

    return np.concatenate((np.nan_to_num(enc_gen), np.nan_to_num(enc_frg))), \
        np.concatenate((np.ones_like(enc_gen[:, 0]), np.zeros_like(enc_frg[:, 0]))), np.array(gen_mean)


def train_svc(x, y):
    c = NuSVC(nu=0.7, gamma=0.01, verbose=True)
    c.fit(x, y)
    return c


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
