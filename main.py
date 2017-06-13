#!/usr/bin/env python

from utils.evaluation.svc import evaluate_svc, prepare_svc_evaluations_csv, get_svc_train_data, train_svc, \
    save_svc_evaluation, save_svc_avg_evaluation, get_svc_evaluation_data
from utils.io import prepare_output_directory
from utils.rnn import get_autoencoder_train_data, load_encoder


def process_models():
    prepare_output_directory()
    prepare_svc_evaluations_csv()

    x, y = get_autoencoder_train_data()
    e = load_encoder(x, y)

    import itertools

    for nu, gamma in itertools.product(range(1, 101), range(1, 101)):
        print('Nu:', nu / 100.0, 'Gamma:', gamma / 100.0, 'F1:', end=' ')
        x, y = get_svc_train_data(e)
        c = train_svc(x, y, nu, gamma)

        for usr_num, (x, y) in enumerate(zip(*get_svc_evaluation_data(e)), 1):
            evl = evaluate_svc(c, x, y, usr_num)
            save_svc_evaluation(evl)
        save_svc_avg_evaluation()


if __name__ == '__main__':
    process_models()
