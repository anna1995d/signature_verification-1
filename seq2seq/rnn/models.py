import keras.backend as K
from keras import layers, losses
from keras.callbacks import EarlyStopping
from keras.layers import Masking, Input, RepeatVector, Dense, Lambda
from keras.layers.wrappers import Bidirectional
from keras.models import Model

from seq2seq.logging import elogger
from seq2seq.rnn.layers import AttentionWithContext
from seq2seq.rnn.logging import rnn_tblogger
from utils.config import CONFIG


class Autoencoder(object):
    def __init__(self):
        self.seq_enc = None
        self.seq_autoenc = None

    def fit(self, x, y):
        self.seq_autoenc.fit(
            x, y,
            epochs=CONFIG.ae_tr_epochs,
            batch_size=CONFIG.ae_btch_sz,
            verbose=CONFIG.verbose,
            callbacks=[elogger, rnn_tblogger(), EarlyStopping(**CONFIG.clbs['early_stopping'])]
        )

    def predict(self, inp):
        return self.seq_enc.predict(inp)

    def save(self, path):
        self.seq_autoenc.save(path)

    def load(self, path):
        self.seq_autoenc.load_weights(path)


# TODO: Add LearningRateScheduler if it is useful
class RecurrentVariationalAutoencoder(Autoencoder):
    def __init__(self, max_len):
        super().__init__()

        cell = getattr(layers, CONFIG.ct)

        # Input
        inp = Input(shape=(None, CONFIG.inp_dim))
        msk = Masking(mask_value=CONFIG.msk_val)(inp)

        # Encoder
        enc = None
        for idx, ln in enumerate(CONFIG.enc_arc):
            c = cell(ln, **CONFIG.ae_lcfg)
            if idx == len(CONFIG.enc_arc) - 1:
                c.return_sequences = False
            enc = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(
                msk if enc is None else enc
            )

        # Latent
        z_mean = Dense(CONFIG.enc_arc[-1])(enc)
        z_log_sigma = Dense(CONFIG.enc_arc[-1])(enc)

        def sampling(args):
            epsilon = K.random_normal(
                shape=(CONFIG.ae_btch_sz, CONFIG.enc_arc[-1]), mean=CONFIG.ltn_mn, stddev=CONFIG.ltn_std
            )
            return args[0] + K.exp(args[1] / 2) * epsilon

        z = Lambda(sampling)([z_mean, z_log_sigma])

        gnr_layers = [RepeatVector(max_len)]

        # Repeat
        rpt = gnr_layers[-1](z)

        # Decoder
        dec = None
        for ln in CONFIG.dec_arc:
            c = cell(ln, **CONFIG.ae_lcfg)
            gnr_layers.append(Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)
            dec = gnr_layers[-1](rpt if dec is None else dec)

        def vae_loss(y_true, y_pred):
            xent_loss = K.sum(getattr(losses, CONFIG.loss_fn)(y_true, y_pred), axis=-1)
            kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return xent_loss + kl_loss

        # Autoencoder
        CONFIG.ae_ccfg['loss'] = [vae_loss]
        self.seq_autoenc = Model(inp, dec)
        self.seq_autoenc.compile(**CONFIG.ae_ccfg)

        # Encoder
        self.seq_enc = Model(inp, z_mean)

        # Generator
        gnr_inp = Input(shape=(CONFIG.enc_arc[-1],))
        gnr = None
        for layer in gnr_layers:
            gnr = layer(gnr_inp if gnr is None else gnr)
        self.seq_gnr = Model(gnr_inp, gnr)


class AttentiveRecurrentAutoencoder(Autoencoder):
    def __init__(self, max_len):
        super().__init__()

        cell = getattr(layers, CONFIG.ct)

        # Input
        inp = Input(shape=(None, CONFIG.inp_dim))
        msk = Masking(mask_value=CONFIG.msk_val)(inp)

        # Encoder
        enc = None
        for ln in CONFIG.enc_arc:
            c = cell(ln, **CONFIG.ae_lcfg)
            enc = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(
                msk if enc is None else enc
            )

        # Attention
        att = AttentionWithContext()(enc)

        # Repeat
        rpt = RepeatVector(max_len)(att)

        # Decoder
        dec = None
        for ln in CONFIG.dec_arc:
            c = cell(ln, **CONFIG.ae_lcfg)
            dec = (Bidirectional(c, merge_mode=CONFIG.bd_merge_mode) if CONFIG.bd_cell_type else c)(
                rpt if dec is None else dec
            )

        # Autoencoder
        self.seq_autoenc = Model(inp, dec)
        self.seq_autoenc.compile(**CONFIG.ae_ccfg)

        # Encoder
        self.seq_enc = Model(inp, att)
