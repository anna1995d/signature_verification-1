#!/usr/bin/env python

from utils import io, rnn, evaluation
from utils.config import CONFIG


def process_model():
    io.prepare_output_directory()

    evaluations = list()
    for fold in range(CONFIG.spt_cnt) if CONFIG.spt_cnt > 0 else [-1]:
        x, y, x_cv, y_cv = rnn.get_autoencoder_train_data(fold)
        encoder = rnn.load_encoder(x, y, x_cv, y_cv, fold if fold != -1 else "")

        x_train, y_train, x_cv, y_cv = evaluation.get_siamese_evaluation_train_data(fold)
        x_test, y_test = evaluation.get_siamese_evaluation_test_data(fold)
        evaluations.append(evaluation.get_optimized_evaluation(
            encoder, x_train, y_train, x_cv, y_cv, x_test, y_test, fold if fold != -1 else ""
        ))
        evaluation.save_evaluation(evaluations)


if __name__ == '__main__':
    process_model()
