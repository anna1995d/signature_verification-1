import keras.backend as K
from keras import layers
from keras.layers import Masking, Input, RepeatVector, Lambda
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping

from seq2seq.logging import blogger, elogger
from seq2seq.rnn.layers import Attention
from seq2seq.rnn.logging import rnn_tblogger
from utils.config import CONFIG


class Autoencoder(object):
    def __init__(self):
        self.seq_enc = None
        self.seq_autoenc = None

    def fit(self, x, y):
        callbacks = [blogger, elogger, rnn_tblogger(), EarlyStopping(monitor='acc', patience=5, verbose=1)]
        self.seq_autoenc.fit(x, y, callbacks=callbacks, **CONFIG.ae_tr)

    def predict(self, inp):
        return self.seq_enc.predict(inp)

    def save(self, path):
        self.seq_autoenc.save(path)

    def load(self, path):
        self.seq_autoenc.load_model(path)


class RecurrentVariationalAutoencoder(Autoencoder):
    def __init__(self, max_len):
        super().__init__()

        cell = getattr(layers, CONFIG.ct)

        # Input
        inp = Input(shape=(None, CONFIG.inp_dim))
        msk = Masking()(inp)

        # Encoder
        enc = None
        for layer in CONFIG.enc_arc:
            enc = Bidirectional(cell(**layer), merge_mode='ave')(msk if enc is None else enc)

        # Latent
        c = cell(CONFIG.enc_arc[-1], **CONFIG.ae_lcfg)
        c.return_sequences = False
        z_mean = Bidirectional(c, merge_mode='ave')(enc)

        c = cell(CONFIG.enc_arc[-1], **CONFIG.ae_lcfg)
        c.return_sequences = False
        z_log_sigma = Bidirectional(c, merge_mode='ave')(enc)

        def sampling(args):
            epsilon = K.random_normal(shape=(1, CONFIG.enc_arc[-1]), mean=0.0, stddev=1.0)
            return args[0] + K.exp(args[1] / 2) * epsilon

        z = Lambda(sampling)([z_mean, z_log_sigma])

        # Repeat
        rpt = RepeatVector(max_len)(z)

        # Decoder
        dec = None
        for layer in CONFIG.dec_arc:
            dec = Bidirectional(cell(**layer), merge_mode='ave')(rpt if dec is None else dec)

        def vae_loss(y_true, y_pred):
            kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return K.sum(K.sum(K.abs(y_true - y_pred), axis=-1), axis=-1) + kl_loss

        # Autoencoder
        self.seq_autoenc = Model(inp, dec)
        self.seq_autoenc.compile(loss=vae_loss, **CONFIG.ae_ccfg)

        # Encoder
        self.seq_enc = Model(inp, z_mean)


class AttentiveRecurrentAutoencoder(Autoencoder):
    def __init__(self, max_len):
        super().__init__()

        cell = getattr(layers, CONFIG.ct)

        # Input
        inp = Input(shape=(None, CONFIG.inp_dim))
        msk = Masking()(inp)

        # Encoder
        enc = None
        for layer in CONFIG.enc_arc:
            enc = Bidirectional(cell(**layer), merge_mode='ave')(msk if enc is None else enc)

        # Attention
        att = Attention()(enc)

        # Repeat
        rpt = RepeatVector(max_len)(att)

        # Decoder
        dec = None
        for layer in CONFIG.dec_arc:
            dec = Bidirectional(cell(**layer), merge_mode='ave')(rpt if dec is None else dec)

        def sum_absolute_error(y_true, y_pred):
            return K.sum(K.abs(y_true - y_pred), axis=-1)

        # Autoencoder
        self.seq_autoenc = Model(inp, dec)
        self.seq_autoenc.compile(loss=sum_absolute_error, **CONFIG.ae_ccfg)

        # Encoder
        self.seq_enc = Model(inp, att)
