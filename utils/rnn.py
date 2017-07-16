import os

import numpy as np
from keras.preprocessing import sequence

from seq2seq.rnn.models import AttentiveRecurrentAutoencoder
from utils.data import DATA
from utils.config import CONFIG


def get_autoencoder_train_data():
    x, y = list(), list()
    for usr_num in range(CONFIG.usr_cnt):
        (gen_x, gen_y) = DATA.get_genuine_combinations(usr_num)
        x.append(sequence.pad_sequences(gen_x, maxlen=DATA.gen_max_len))
        y.append(sequence.pad_sequences(gen_y, maxlen=DATA.gen_max_len))
    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)


def load_encoder(x, y):
    arae = AttentiveRecurrentAutoencoder(max_len=x.shape[1])
    arae.fit(x, y)
    arae.save(path=os.path.join(CONFIG.out_dir, 'autoencoder.dat'))
    return arae


def get_encoded_data(e, non_enc):
    return e.predict(inp=sequence.pad_sequences(non_enc))
