#!/usr/bin/env python

from utils.evaluation import svc, knc, data, utils
from utils import io, rnn


def process_model():
    io.prepare_output_directory()

    x, y = rnn.get_autoencoder_train_data()
    e = rnn.load_encoder(x, y)

    x_tr, y_tr = data.get_evaluation_train_data(e)
    x_cv, y_cv = data.get_evaluation_cross_validation_data(e)
    x_ts, y_ts = data.get_evaluation_test_data(e)

    utils.save_evaluation(svc.get_optimized_evaluation(x_tr, y_tr, x_cv, y_cv, x_ts, y_ts), 'svc')
    utils.save_evaluation(knc.get_optimized_evaluation(x_tr, y_tr, x_ts, y_ts), 'knc')


if __name__ == '__main__':
    process_model()
