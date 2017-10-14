#!/usr/bin/env python

from utils import io, rnn, evaluation
from utils.config import CONFIG


def process_model():
    io.prepare_directories()

    evaluations = list()
    for fold in range(CONFIG.spt_cnt) if CONFIG.spt_cnt > 0 else [-1]:
        x, y, x_cv, y_cv = rnn.get_autoencoder_data(fold)
        encoder = rnn.get_encoder(x, y, x_cv, y_cv, fold if fold != -1 else "")

        x, y, x_cv, y_cv, x_ts, y_ts = evaluation.get_siamese_data(fold)
        evaluations.append(evaluation.get_optimized_evaluation(
            encoder, x, y, x_cv, y_cv, x_ts, y_ts, fold if fold != -1 else ""
        ))
        evaluation.save_evaluation(evaluations)


if __name__ == '__main__':
    process_model()
