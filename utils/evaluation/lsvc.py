import csv
import os

import numpy as np
from sklearn.metrics import classification_report

from seq2seq.models import LinearSVC
from utils.config import CONFIG


def prepare_lsvc_output_directory(out_dir):
    lsvcs_dir = os.path.join(out_dir, 'linear_svcs')
    if not os.path.exists(lsvcs_dir):
        os.mkdir(lsvcs_dir)

    return lsvcs_dir


def get_lsvc_train_data(enc_gen, enc_frg):
    return np.concatenate((enc_gen, enc_frg)), \
           np.concatenate((np.ones_like(enc_gen[:, 0]), np.zeros_like(enc_frg[:, 0])))


def train_lsvc(x, y, usr_num, lsvcs_dir):
    c = LinearSVC()
    c.fit(x, y)
    c.save(path=os.path.join(lsvcs_dir, CONFIG.mdl_save_temp.format(usr_num=usr_num)))
    cr = list(map(float, classification_report(y_true=y, y_pred=c.predict(x)).split('\n')[-2].split()[3:6]))
    return {
        CONFIG.lsvc_csv_fns[0]: usr_num,
        CONFIG.lsvc_csv_fns[3]: cr[0],
        CONFIG.lsvc_csv_fns[4]: cr[1],
        CONFIG.lsvc_csv_fns[5]: cr[2]
    }


def evaluate_lsvc(usr_num, enc_gen, enc_frg, lsvcs_dir):
    x, y = get_lsvc_train_data(enc_gen, enc_frg)
    return train_lsvc(x, y, usr_num, lsvcs_dir)


def prepare_lsvc_evaluations_csv(lsvcs_dir):
    with open(os.path.join(lsvcs_dir, 'evaluations.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.lsvc_csv_fns)
        w.writeheader()


def save_lsvc_evaluation(evl, lsvcs_dir):
    with open(os.path.join(lsvcs_dir, 'evaluations.csv'), 'a') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.lsvc_csv_fns)
        w.writerow(evl)


def save_lsvc_avg_evaluation(lsvcs_dir):
    with open(os.path.join(lsvcs_dir, 'evaluations.csv'), 'r') as f:
        avg = {
            CONFIG.lsvc_csv_fns[0]: 'AVG'
        }
        rows = [r for r in csv.DictReader(f, fieldnames=CONFIG.lsvc_csv_fns)][1:]
        avg.update({
            CONFIG.lsvc_csv_fns[i]: np.mean(
                [float(r[CONFIG.lsvc_csv_fns[i]]) for r in rows]
            ) for i in range(1, len(CONFIG.lsvc_csv_fns))
        })

    with open(os.path.join(lsvcs_dir, 'evaluations.csv'), 'a') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.lsvc_csv_fns)
        w.writerow(avg)
