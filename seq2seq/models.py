import keras.backend as K
from keras import layers, losses
from keras.callbacks import EarlyStopping
from keras.layers import Masking, Input, RepeatVector, Lambda, Dense
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


class RecurrentVariationalAutoencoder(Autoencoder):
    def __init__(self, max_len):
        super().__init__()

        cell = getattr(layers, CONFIG.ct)

        # Input
        inp = Input(shape=(None, CONFIG.inp_dim * CONFIG.win_sze))
        msk = Masking()(inp)

        # Encoder
        enc = None
        for layer in CONFIG.enc_arc:
            enc = Bidirectional(cell(**layer), merge_mode='ave')(msk if enc is None else enc)

        # Latent
        z_mean = Dense(units=CONFIG.enc_arc[-1]['units'])(enc)
        z_log_sigma = Dense(units=CONFIG.enc_arc[-1]['units'])(enc)

        def sampling(args):
            epsilon = K.random_normal(shape=(1, CONFIG.enc_arc[-1]['units']), mean=0.0, stddev=1.0)
            return args[0] + K.exp(args[1] / 2) * epsilon

        z = Lambda(sampling, output_shape=(CONFIG.enc_arc[-1]['units'],))([z_mean, z_log_sigma])

        # Repeat
        rpt = RepeatVector(max_len)(z)

        # Decoder
        dec = None
        for layer in CONFIG.dec_arc:
            dec = Bidirectional(cell(**layer), merge_mode='ave')(rpt if dec is None else dec)

        loss_fn = CONFIG.ae_ccfg.pop('loss')

        def vae_loss(y_true, y_pred):
            xent_loss = K.sum(getattr(losses, loss_fn)(y_true, y_pred), axis=-1)
            kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return xent_loss + kl_loss

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
        inp = Input(shape=(None, CONFIG.inp_dim * CONFIG.win_sze))
        msk = Masking()(inp)

        # Encoder
        enc = None
        for layer in CONFIG.enc_arc:
            enc = Bidirectional(cell(**layer), merge_mode='ave')(msk if enc is None else enc)

        # Attention
        att = AttentionWithContext()(enc)

        # Repeat
        rpt = RepeatVector(max_len)(att)

        # Decoder
        dec = None
        for layer in CONFIG.dec_arc:
            dec = Bidirectional(cell(**layer), merge_mode='ave')(rpt if dec is None else dec)

        # Autoencoder
        self.seq_autoenc = Model(inp, dec)
        self.seq_autoenc.compile(**CONFIG.ae_ccfg)

        # Encoder
        self.seq_enc = Model(inp, att)
