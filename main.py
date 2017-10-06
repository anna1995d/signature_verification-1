#!/usr/bin/env python

from utils import io, rnn, evaluation
from utils.config import CONFIG


def process_model():
    io.prepare_output_directory()

    evaluations = list()
    for fold in range(CONFIG.spt_cnt):
        x, y = rnn.get_autoencoder_train_data(fold)
        encoder = rnn.load_encoder(x, y, fold)

        x_train, y_train = evaluation.get_siamese_evaluation_train_data(encoder, fold)
        x_test, y_test = evaluation.get_siamese_evaluation_test_data(encoder, fold)
        evaluations.append(evaluation.get_optimized_evaluation(x_train, y_train, x_test, y_test, fold))
    evaluation.save_evaluation(evaluations)


if __name__ == '__main__':
    process_model()
