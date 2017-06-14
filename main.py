#!/usr/bin/env python

from utils.evaluation.svc import evaluate_svc, prepare_svc_evaluations_csv, get_svc_train_data, train_svc, \
    save_svc_evaluation, get_svc_evaluation_data
from utils.io import prepare_output_directory
from utils.rnn import get_autoencoder_train_data, load_encoder


def process_models():
    prepare_output_directory()

    x, y = get_autoencoder_train_data()
    e = load_encoder(x, y)

    x, y = get_svc_train_data(e)
    c = train_svc(x, y)

    prepare_svc_evaluations_csv()
    x, y = get_svc_evaluation_data(e)
    evl = evaluate_svc(c, x, y, 'AVG')
    save_svc_evaluation(evl)


if __name__ == '__main__':
    process_models()
