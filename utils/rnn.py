import os

import numpy as np
from keras.preprocessing import sequence

from seq2seq.models import AttentiveRecurrentAutoencoder
from utils.config import CONFIG
from utils.data import DATA


def get_autoencoder_train_data(fold):
    x, y = list(), list()
    for writer in range(CONFIG.wrt_cnt):
        if writer // (CONFIG.wrt_cnt // CONFIG.spt_cnt) == fold:
            continue

        (gen_x, gen_y) = DATA.get_train_data(writer)
        if gen_x is None and gen_y is None:
            continue
        x.append(sequence.pad_sequences(gen_x, maxlen=DATA.gen_max_len))
        y.append(sequence.pad_sequences(gen_y, maxlen=DATA.gen_max_len))
    return np.concatenate(x), np.concatenate(y)


def load_encoder(x, y, fold):
    attentive_recurrent_autoencoder = AttentiveRecurrentAutoencoder(x.shape[1], fold)
    if CONFIG.ae_md == 'train':
        attentive_recurrent_autoencoder.fit(x, y)
        attentive_recurrent_autoencoder.save(os.path.join(CONFIG.out_dir, 'autoencoder_fold{}.hdf5').format(fold))
    else:
        attentive_recurrent_autoencoder.load(os.path.join(CONFIG.out_dir, 'autoencoder_fold{}.hdf5').format(fold))
    return attentive_recurrent_autoencoder


def get_encoded_data(encoder, original):
    return encoder.predict(sequence.pad_sequences(original))
