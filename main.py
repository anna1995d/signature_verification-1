#!/usr/bin/env python

from utils import io, rnn, evaluation


def process_model():
    io.prepare_output_directory()

    x, y = rnn.get_autoencoder_train_data()
    encoder = rnn.load_encoder(x, y)

    x_train, y_train = evaluation.get_siamese_evaluation_train_data(encoder)
    x_test, y_test = evaluation.get_siamese_evaluation_test_data(encoder)
    evl = evaluation.get_optimized_evaluation(x_train, y_train, x_test, y_test)
    evaluation.save_evaluation(evl)


if __name__ == '__main__':
    process_model()
