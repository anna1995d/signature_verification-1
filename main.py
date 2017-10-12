#!/usr/bin/env python

from utils import io, rnn, evaluation
from utils.config import CONFIG


def process_model():
    io.prepare_directories()

    evaluations = list()
    for fold in range(CONFIG.spt_cnt) if CONFIG.spt_cnt > 0 else [-1]:
        tr_generator, cv_generator = rnn.get_autoencoder_data_generators(fold)
        encoder = rnn.get_encoder(tr_generator, cv_generator, fold if fold != -1 else "")

        tr_generator, cv_generator, ts_generator, y_true = evaluation.get_siamese_data_generators(fold)
        evaluations.append(evaluation.get_optimized_evaluation(
            encoder, tr_generator, cv_generator, ts_generator, y_true, fold if fold != -1 else ""
        ))
        evaluation.save_evaluation(evaluations)

    io.clean_directories()


if __name__ == '__main__':
    process_model()
