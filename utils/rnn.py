import os

from keras.preprocessing import sequence

from seq2seq.rnn.models import AttentiveRecurrentAutoencoder
from utils.config import CONFIG


def get_autoencoder_train_data(data, usr_num):
    (gen_x, gen_y) = data.get_genuine_combinations(usr_num, CONFIG.ae_smp_cnt)
    return sequence.pad_sequences(gen_x, value=CONFIG.msk_val), \
        sequence.pad_sequences(gen_y, value=CONFIG.msk_val), \
        sequence.pad_sequences(data.gen[usr_num][:CONFIG.ae_smp_cnt], value=CONFIG.msk_val), \
        sequence.pad_sequences(data.frg[usr_num][:CONFIG.ae_smp_cnt], value=CONFIG.msk_val), \
        sequence.pad_sequences(data.gen[usr_num][CONFIG.ae_smp_cnt:], value=CONFIG.msk_val), \
        sequence.pad_sequences(data.frg[usr_num][CONFIG.ae_smp_cnt:], value=CONFIG.msk_val)


def load_encoder(x, y, usr_num):
    arae = AttentiveRecurrentAutoencoder(max_len=x.shape[1])
    arae.fit(x, y, usr_num=usr_num)
    arae.save(path=os.path.join(CONFIG.aes_dir, CONFIG.mdl_save_temp.format(usr_num=usr_num)))
    return arae


def get_encoded_data(e, non_enc):
    return e.predict(inp=sequence.pad_sequences(non_enc, value=CONFIG.msk_val))
