import os

import numpy as np
from keras.preprocessing import sequence

from seq2seq.rnn.models import AttentiveRecurrentAutoencoder
from utils.data import DATA
from utils.config import CONFIG


def get_autoencoder_evaluation_data(e, m):
    x, y = list(), list()
    for usr_num in range(CONFIG.usr_cnt):
        enc_gen, enc_frg = get_encoded_data(e, DATA.gen[usr_num][CONFIG.ae_smp_cnt:]) - m[usr_num], \
                           get_encoded_data(e, DATA.frg[usr_num][CONFIG.ae_smp_cnt:]) - m[usr_num]
        x.append(np.concatenate((np.nan_to_num(enc_gen), np.nan_to_num(enc_frg))))
        y.append(np.concatenate((np.ones_like(enc_gen[:, 0]), np.zeros_like(enc_frg[:, 0]))))
    return x, y


def get_autoencoder_train_data():
    x, y = list(), list()
    for usr_num in range(CONFIG.usr_cnt):
        (gen_x, gen_y) = DATA.get_genuine_combinations(usr_num, CONFIG.ae_smp_cnt)
        x.append(sequence.pad_sequences(gen_x, value=CONFIG.msk_val, maxlen=DATA.gen_max_len))
        y.append(sequence.pad_sequences(gen_y, value=CONFIG.msk_val, maxlen=DATA.gen_max_len))
    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)


def load_encoder(x, y):
    arae = AttentiveRecurrentAutoencoder(max_len=x.shape[1])
    arae.fit(x, y)
    arae.save(path=os.path.join(CONFIG.out_dir, 'autoencoder.dat'))
    return arae


def get_encoded_data(e, non_enc):
    return e.predict(inp=sequence.pad_sequences(non_enc, value=CONFIG.msk_val))
