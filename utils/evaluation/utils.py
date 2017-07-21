import csv
import os

from utils.config import CONFIG


def save_evaluation(evl, mode):
    with open(os.path.join(CONFIG.out_dir, '{mode}_evaluations.csv').format(mode=mode), 'w') as f:
        w = csv.DictWriter(f, fieldnames=CONFIG.csv[mode])
        w.writeheader()
        w.writerow(evl)
