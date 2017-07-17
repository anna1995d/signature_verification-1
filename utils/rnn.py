import os

from keras.preprocessing import sequence

from seq2seq.rnn.models import AttentiveRecurrentAutoencoder
from utils.config import CONFIG
from utils.data import DATA


def get_autoencoders_train_data():
    x, y = list(), list()
    for usr_num in range(CONFIG.usr_cnt):
        (gen_x, gen_y) = DATA.get_genuine_combinations(usr_num)
        x.append(sequence.pad_sequences(gen_x, maxlen=DATA.gen_max_len[usr_num]))
        y.append(sequence.pad_sequences(gen_y, maxlen=DATA.gen_max_len[usr_num]))
    return x, y


def load_encoders(x, y):
    araes = list()
    for usr_num in range(CONFIG.usr_cnt):
        arae = AttentiveRecurrentAutoencoder(max_len=x[usr_num].shape[1])
        arae.fit(x[usr_num], y[usr_num])
        arae.save(path=os.path.join(CONFIG.out_dir, 'autoencoder_U{usr_num}.h5'.format(usr_num=usr_num)))
        araes.append(arae)
    return araes


def get_encoded_data(e, non_enc):
    return e.predict(inp=sequence.pad_sequences(non_enc))
