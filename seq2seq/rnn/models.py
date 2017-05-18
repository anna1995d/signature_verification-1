from keras import layers
from keras.layers import Masking, Input, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.models import Model

from seq2seq.logging import elogger
from seq2seq.rnn.layers import AttentionWithContext
from seq2seq.rnn.logging import rnn_tblogger
from utils.config import CONFIG


# TODO: Add EarlyStopping Callback
# TODO: Add LearningRateScheduler if it is useful
class Autoencoder(object):
    def __init__(self, max_len):
        cell = getattr(layers, CONFIG.cell_type)

        # Input
        inp = Input(shape=(None, CONFIG.inp_dim), name='input')
        msk = Masking(mask_value=CONFIG.msk_val, name='mask')(inp)

        # Encoder
        c = cell(CONFIG.enc_arc[0], **CONFIG.ae_lcfg, name='encoder_{index}'.format(index=0))
        enc = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(msk)
        for i, ln in enumerate(CONFIG.enc_arc[1:], 1):
            c = cell(ln, **CONFIG.ae_lcfg, name='encoder_{index}'.format(index=i))
            enc = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(enc)

        # Attention
        att = AttentionWithContext()(enc)

        # Decoder
        rpt_vec = RepeatVector(max_len)(att)

        c = cell(CONFIG.dec_arc[0], **CONFIG.ae_lcfg, name='decoder_{index}'.format(index=0))
        dec = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(rpt_vec)
        for i, ln in enumerate(CONFIG.dec_arc[1:], 1):
            c = cell(ln, **CONFIG.ae_lcfg, name='decoder_{index}'.format(index=i))
            dec = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(dec)

        self.seq_autoenc = Model(inp, dec)
        self.seq_autoenc.compile(**CONFIG.ae_ccfg)

    def fit(self, x, y, usr_num):
        self.seq_autoenc.fit(
            x, y,
            epochs=CONFIG.ae_tr_epochs,
            batch_size=CONFIG.ae_btch_sz,
            verbose=CONFIG.verbose,
            callbacks=[elogger, rnn_tblogger(usr_num)]
        )

    def save(self, path):
        self.seq_autoenc.save(path)


class Encoder(object):
    def __init__(self):
        cell = getattr(layers, CONFIG.cell_type)

        # Input
        inp = Input(shape=(None, CONFIG.inp_dim), name='input')
        msk = Masking(mask_value=CONFIG.msk_val, name='mask')(inp)

        # Encoder
        c = cell(CONFIG.enc_arc[0], **CONFIG.ae_lcfg, name='encoder_{index}'.format(index=0))
        c.return_sequences = (0 != len(CONFIG.enc_arc) - 1)
        enc = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(msk)
        for i, ln in enumerate(CONFIG.enc_arc[1:], 1):
            c = cell(ln, **CONFIG.ae_lcfg, name='encoder_{index}'.format(index=i))
            c.return_sequences = (i != len(CONFIG.enc_arc) - 1)
            enc = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(enc)

        self.seq_enc = Model(inp, enc)
        self.seq_enc.compile(**CONFIG.ae_ccfg)

    def predict(self, inp):
        return self.seq_enc.predict(inp)

    def load(self, path):
        self.seq_enc.load_weights(path, by_name=True)
