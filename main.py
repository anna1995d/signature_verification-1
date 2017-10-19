#!/usr/bin/env python


def process():
    from utils import io, rnn, evaluation
    from utils.config import CONFIG

    io.prepare_directories()

    evaluations = list()
    for fold in range(CONFIG.spt_cnt) if CONFIG.spt_cnt > 0 else [-1]:
        x, y, x_cv, y_cv = rnn.get_autoencoder_data(fold)
        encoder = rnn.get_encoder(x, y, x_cv, y_cv, fold if fold != -1 else "")

        x, y, x_cv, y_cv, x_ts_1, y_ts_1, x_ts_2, y_ts_2 = evaluation.get_siamese_data(encoder, fold)
        evaluations.append(evaluation.get_evaluation(
            x, y, x_cv, y_cv, x_ts_1, y_ts_1, x_ts_2, y_ts_2, fold if fold != -1 else ""
        ))
        evaluation.save_evaluation(evaluations)


if __name__ == '__main__':
    import os
    import numpy as np

    np.random.seed(os.getenv('SEED'))

    process()
