import numpy as np
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import NuSVC

from utils.config import CONFIG


def _scorer(y, y_pred):
    scores = list(map(float, classification_report(y_true=y, y_pred=y_pred).split('\n')[-2].split()[3:6]))
    return scores[2]


def get_optimized_evaluation(x_tr, y_tr, x_ts, y_ts):
    x, y = np.concatenate([x_tr, x_tr]), np.concatenate([y_tr, y_tr])
    estimator = NuSVC()
    param_grid = [{
        'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
        'nu': np.arange(start=0.010, stop=0.850, step=0.001, dtype=np.float64),
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
    cv = PredefinedSplit(test_fold=np.concatenate([np.ones_like(x_tr[:, 0]) * (-1), np.zeros_like(x_tr[:, 0])]))
    c = GridSearchCV(
        estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, return_train_score=False, n_jobs=-1
    )
    c.fit(x, y)

    scores = list(map(
        float, classification_report(y_true=y_ts, y_pred=c.best_estimator_.predict(x_ts)).split('\n')[-2].split()[3:6]
    ))

    return {
        CONFIG.csv['svc'][0]: c.best_params_['kernel'],
        CONFIG.csv['svc'][1]: c.best_params_['nu'],
        CONFIG.csv['svc'][2]: c.best_params_['gamma'],
        CONFIG.csv['svc'][3]: scores[2]
    }
