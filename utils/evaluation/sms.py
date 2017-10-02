import os

import numpy as np
from sklearn.metrics import classification_report

from seq2seq.models import SiameseClassifier
from utils.config import CONFIG


def get_optimized_evaluation(x_tr, y_tr, x_ts, y_ts):
    sms = SiameseClassifier()
    sms.fit(x_tr, y_tr)
    sms.save(path=os.path.join(CONFIG.out_dir, 'siamese.hdf5'))

    scores = list(map(float, classification_report(
        y_true=y_ts, y_pred=(sms.predict(x_ts) >= 0.5).astype(np.int32), digits=CONFIG.clf_rpt_dgt
    ).split('\n')[-2].split()[3:6]))

    return {
        CONFIG.csv['sms'][0]: scores[0],
        CONFIG.csv['sms'][1]: scores[1],
        CONFIG.csv['sms'][2]: scores[2]
    }
