import os

import numpy as np
from keras.preprocessing import sequence

from seq2seq.models import AttentiveRecurrentAutoencoder
from utils.config import CONFIG
from utils.data import DATA


def get_autoencoder_train_data(fold):
    x, y, x_cv, y_cv = list(), list(), list(), list()
    for writer in range(CONFIG.wrt_cnt):
        (gen_x, gen_y) = DATA.get_train_data(writer)
        if gen_x is None and gen_y is None:
            continue
        if 0 <= fold == writer // (CONFIG.wrt_cnt // CONFIG.spt_cnt):
            x_cv.append(sequence.pad_sequences(gen_x, maxlen=DATA.max_len))
            y_cv.append(sequence.pad_sequences(gen_y, maxlen=DATA.max_len))
        elif fold < 0 and writer >= CONFIG.tr_wrt_cnt:
            x.append(sequence.pad_sequences(gen_x[:CONFIG.ref_smp_cnt], maxlen=DATA.max_len))
            y.append(sequence.pad_sequences(gen_y[:CONFIG.ref_smp_cnt], maxlen=DATA.max_len))

            x_cv.append(sequence.pad_sequences(gen_x[CONFIG.ref_smp_cnt:], maxlen=DATA.max_len))
            y_cv.append(sequence.pad_sequences(gen_y[CONFIG.ref_smp_cnt:], maxlen=DATA.max_len))
        else:
            x.append(sequence.pad_sequences(gen_x, maxlen=DATA.max_len))
            y.append(sequence.pad_sequences(gen_y, maxlen=DATA.max_len))

    x = np.concatenate(x)
    y = np.concatenate(y)
    x_cv = np.concatenate(x_cv)
    y_cv = np.concatenate(y_cv)

    return x, y, x_cv, y_cv


def load_encoder(x, y, x_cv, y_cv, fold):
    attentive_recurrent_autoencoder = AttentiveRecurrentAutoencoder(x.shape[1], fold)
    if CONFIG.ae_md == 'train':
        attentive_recurrent_autoencoder.fit(x, y, x_cv, y_cv)
        attentive_recurrent_autoencoder.save(os.path.join(CONFIG.out_dir, 'autoencoder_fold{}.hdf5').format(fold))
    else:
        attentive_recurrent_autoencoder.load(os.path.join(CONFIG.out_dir, 'autoencoder_fold{}.hdf5').format(fold))
    return attentive_recurrent_autoencoder


def get_encoded_data(encoder, original):
    return encoder.predict(sequence.pad_sequences(original))
