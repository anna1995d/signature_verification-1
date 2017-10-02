from sklearn.metrics import classification_report
from sklearn.neighbors.classification import KNeighborsClassifier

from utils.config import CONFIG


def get_optimized_evaluation(x_tr, y_tr, x_ts, y_ts):
    knc = KNeighborsClassifier(n_jobs=-1, **CONFIG.knc)
    knc.fit(x_tr, y_tr)

    scores = list(map(float, classification_report(
        y_true=y_ts, y_pred=knc.predict(x_ts), digits=CONFIG.clf_rpt_dgt
    ).split('\n')[-2].split()[3:6]))

    return {
        CONFIG.csv['knc'][0]: scores[0],
        CONFIG.csv['knc'][1]: scores[1],
        CONFIG.csv['knc'][2]: scores[2]
    }
