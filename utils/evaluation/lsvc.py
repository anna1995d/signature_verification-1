import csv
import os

import numpy as np
from sklearn.metrics import classification_report

from seq2seq.models import LinearSVC
from utils.config import CONFIG
from utils.data import DATA


def get_lsvc_train_data(enc_gen, enc_frg):
    return np.concatenate((np.nan_to_num(enc_gen), np.nan_to_num(enc_frg))), \
           np.concatenate((np.ones_like(enc_gen[:, 0]), np.zeros_like(enc_frg[:, 0])))


def train_lsvc(x, y):
    c = LinearSVC()
    c.fit(x, y)
    return c


def evaluate_lsvc(c, x, y, usr_num):
    cr = list(map(float, classification_report(y_true=y, y_pred=c.predict(x)).split('\n')[-2].split()[3:6]))
    return {
        CONFIG.lsvc_csv_fns[0]: usr_num,
        CONFIG.lsvc_csv_fns[1]: np.mean(list(map(len, DATA.gen[usr_num - 1]))),
        CONFIG.lsvc_csv_fns[2]: np.mean(list(map(len, DATA.frg[usr_num - 1]))),
        CONFIG.lsvc_csv_fns[3]: cr[0],
        CONFIG.lsvc_csv_fns[4]: cr[1],
        CONFIG.lsvc_csv_fns[5]: cr[2]
    }


def prepare_lsvc_evaluations_csv():
    with open(os.path.join(CONFIG.out_dir, 'lsvc_evaluations.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.lsvc_csv_fns)
        w.writeheader()


def save_lsvc_evaluation(evl):
    with open(os.path.join(CONFIG.out_dir, 'lsvc_evaluations.csv'), 'a') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.lsvc_csv_fns)
        w.writerow(evl)


def save_lsvc_avg_evaluation():
    with open(os.path.join(CONFIG.out_dir, 'lsvc_evaluations.csv'), 'r') as f:
        avg = {
            CONFIG.lsvc_csv_fns[0]: 'AVG'
        }
        rows = [r for r in csv.DictReader(f, fieldnames=CONFIG.lsvc_csv_fns)][1:]
        avg.update({
            CONFIG.lsvc_csv_fns[i]: np.mean(
                [float(r[CONFIG.lsvc_csv_fns[i]]) for r in rows]
            ) for i in range(1, len(CONFIG.lsvc_csv_fns))
        })
    save_lsvc_evaluation(avg)
