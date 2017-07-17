#!/usr/bin/env python

from utils.evaluation.svc import prepare_svc_evaluations_csv, get_svc_train_data, get_optimized_svc_evaluation, \
    save_svc_evaluation, get_svc_cross_validation_data, get_svc_test_data
from utils.io import prepare_output_directory
from utils.rnn import get_autoencoders_train_data, load_encoders


def process_model():
    prepare_output_directory()

    x, y = get_autoencoders_train_data()
    es = load_encoders(x, y)

    x_train, y_train = get_svc_train_data(es)
    x_cv, y_cv = get_svc_cross_validation_data(es)
    x_ts, y_ts = get_svc_test_data(es)
    evl = get_optimized_svc_evaluation(x_train, y_train, x_cv, y_cv, x_ts, y_ts)

    prepare_svc_evaluations_csv()
    save_svc_evaluation(evl)


if __name__ == '__main__':
    process_model()
