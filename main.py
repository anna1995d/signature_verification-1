#!/usr/bin/env python

from utils import io, rnn
from utils.evaluation import knc, sms, svc, data, utils


def process_model():
    io.prepare_output_directory()

    x, y = rnn.get_autoencoder_train_data()
    encoder = rnn.load_encoder(x, y)

    # KNN and SVM
    x_train, y_train = data.get_evaluation_train_data(encoder)
    x_test, y_test = data.get_evaluation_test_data(encoder)
    utils.save_evaluation(knc.get_optimized_evaluation(x_train, y_train, x_test, y_test), 'knc')
    utils.save_evaluation(svc.get_optimized_evaluation(x_train, y_train, x_test, y_test), 'svc')

    # Siamese
    x_train, y_train = data.get_siamese_evaluation_train_data(encoder)
    x_test, y_test = data.get_siamese_evaluation_test_data(encoder)
    utils.save_evaluation(sms.get_optimized_evaluation(x_train, y_train, x_test, y_test), 'sms')


if __name__ == '__main__':
    process_model()
