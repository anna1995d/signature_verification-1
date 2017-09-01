from keras import layers
from keras.callbacks import EarlyStopping
from keras.layers import Masking, Input, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.models import Model

from seq2seq.layers import AttentionWithContext
from seq2seq.logging import elogger
from utils.config import CONFIG


class Autoencoder(object):
    def __init__(self):
        self.seq_enc = None
        self.seq_autoenc = None

    def fit(self, x, y):
        callbacks = [elogger, EarlyStopping(**CONFIG.clbs['early_stopping'])]
        self.seq_autoenc.fit(x, y, callbacks=callbacks, **CONFIG.ae_tr)

    def predict(self, inp):
        return self.seq_enc.predict(inp)

    def save(self, path):
        self.seq_autoenc.save_weights(path)

    def load(self, path):
        self.seq_autoenc.load_weights(path)


class AttentiveRecurrentAutoencoder(Autoencoder):
    def __init__(self, max_len):
        super().__init__()

        cell = getattr(layers, CONFIG.ct)

        # Input
        inp = Input(shape=(None, CONFIG.inp_dim * CONFIG.win_sze))
        msk = Masking()(inp)

        # Encoder
        enc = None
        for layer in CONFIG.enc_arc:
            merge_mode = layer.pop('merge_mode')
            enc = Bidirectional(cell(**layer), merge_mode=merge_mode)(msk if enc is None else enc)
            layer['merge_mode'] = merge_mode

        # Attention
        att = AttentionWithContext()(enc)

        # Repeat
        rpt = RepeatVector(max_len)(att)

        # Decoder
        dec = None
        for layer in CONFIG.dec_arc:
            merge_mode = layer.pop('merge_mode')
            dec = Bidirectional(cell(**layer), merge_mode=merge_mode)(rpt if dec is None else dec)
            layer['merge_mode'] = merge_mode

        # Autoencoder
        self.seq_autoenc = Model(inp, dec)
        self.seq_autoenc.compile(**CONFIG.ae_ccfg)

        # Encoder
        self.seq_enc = Model(inp, att)
