import os

from keras import layers
from keras.preprocessing import sequence

from seq2seq.rnn.models import Autoencoder, Encoder
from utils.config import CONFIG


def get_autoencoder_train_data(data, usr_num):
    (gen_x, gen_y) = data.get_genuine_combinations(usr_num, CONFIG.ae_smp_cnt)
    return sequence.pad_sequences(gen_x, value=CONFIG.msk_val), \
        sequence.pad_sequences(gen_y, value=CONFIG.msk_val), \
        sequence.pad_sequences(data.gen[usr_num][:CONFIG.ae_smp_cnt], value=CONFIG.msk_val), \
        sequence.pad_sequences(data.gen[usr_num][CONFIG.ae_smp_cnt:], value=CONFIG.msk_val), \
        sequence.pad_sequences(data.frg[usr_num][CONFIG.ae_smp_cnt:], value=CONFIG.msk_val)


def load_encoder(x, y, usr_num):
    def train_autoencoder():
        ae = Autoencoder(
            cell=getattr(layers, CONFIG.cell_type),
            bidir=CONFIG.bd_cell_type,
            bidir_mrgm=CONFIG.bd_merge_mode,
            inp_dim=CONFIG.inp_dim,
            max_len=x.shape[1],
            earc=CONFIG.enc_arc,
            darc=CONFIG.dec_arc,
            msk_val=CONFIG.msk_val,
            ccfg=CONFIG.ae_ccfg,
            lcfg=CONFIG.ae_lcfg
        )
        ae.fit(x, y, epochs=CONFIG.ae_tr_epochs, batch_size=CONFIG.ae_btch_sz, verbose=CONFIG.verbose, usr_num=usr_num)
        ae.save(path=os.path.join(CONFIG.aes_dir, CONFIG.mdl_save_temp.format(usr_num=usr_num)))

    train_autoencoder()
    e = Encoder(
        cell=getattr(layers, CONFIG.cell_type),
        bidir=CONFIG.bd_cell_type,
        bidir_mrgm=CONFIG.bd_merge_mode,
        inp_dim=CONFIG.inp_dim,
        earc=CONFIG.enc_arc,
        msk_val=CONFIG.msk_val,
        ccfg=CONFIG.ae_ccfg,
        lcfg=CONFIG.ae_lcfg
    )
    e.load(path=os.path.join(CONFIG.aes_dir, CONFIG.mdl_save_temp.format(usr_num=usr_num)))
    return e


def get_encoded_data(e, non_enc):
    return e.predict(inp=sequence.pad_sequences(non_enc, value=CONFIG.msk_val))
