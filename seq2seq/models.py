import os

from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Masking, Input, RepeatVector, Dropout, Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model

from seq2seq.layers import AttentionWithContext
from seq2seq.logging import epoch_logger
from utils.config import CONFIG


class CustomModel(object):
    def __init__(self, train_config, model, predictor=None, early_stopping=None, model_checkpoint=None):
        self.train_config = train_config
        self.model = model
        self.predictor = predictor
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Function __call__ is not implemented for class {}!'.format(self.__class__))

    def build_model(self, *args, **kwargs):
        raise NotImplementedError('Function build_model is not implemented for class {}!'.format(self.__class__))

    def fit(self, x, y):
        callbacks = [epoch_logger]
        if self.early_stopping is not None:
            callbacks.append(EarlyStopping(**self.early_stopping))
        if self.model_checkpoint is not None:
            callbacks.append(ModelCheckpoint(os.path.join(CONFIG.out_dir, self.model_checkpoint)))
        self.model.fit(x, y, callbacks=callbacks, **self.train_config)

    def predict(self, x):
        return self.model.predict(x) if self.predictor is None else self.predictor.predict(x)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


class AttentiveRecurrentAutoencoder(CustomModel):
    def __init__(self, max_len, fold):
        seq_autoencoder, seq_encoder = self.build_model(max_len)
        super().__init__(
            CONFIG.ae_tr, seq_autoencoder, seq_encoder,
            CONFIG.ae_clbs['early_stopping'], 'autoencoder_checkpoint_fold{}.hdf5'.format(fold)
        )

    def __call__(self, max_len):
        return self.build_model(max_len)[0]

    def build_model(self, max_len):
        cell = getattr(layers, CONFIG.ct)

        # Input
        network_input = Input(shape=(None, CONFIG.ftr * CONFIG.win_sze))
        mask = Masking()(network_input)

        # Encoder
        encoder = None
        for layer in CONFIG.enc_arc:
            merge_mode = layer.pop('merge_mode')
            encoder = Bidirectional(cell(**layer), merge_mode=merge_mode)(mask if encoder is None else encoder)
            layer['merge_mode'] = merge_mode
        encoder = Dropout(CONFIG.ae_drp)(mask if encoder is None else encoder)

        # Attention
        attention = AttentionWithContext()(encoder)

        # Repeat
        repeat = RepeatVector(max_len)(attention)

        # Decoder
        decoder = None
        for layer in CONFIG.dec_arc:
            merge_mode = layer.pop('merge_mode')
            decoder = Bidirectional(cell(**layer), merge_mode=merge_mode)(repeat if decoder is None else decoder)
            layer['merge_mode'] = merge_mode

        # Autoencoder
        seq_autoencoder = Model(network_input, decoder)
        seq_autoencoder.summary()
        seq_autoencoder.compile(**CONFIG.ae_ccfg)

        # Encoder
        seq_encoder = Model(network_input, attention)

        return seq_autoencoder, seq_encoder


class SiameseClassifier(CustomModel):
    def __init__(self, fold):
        siamese = self.build_model()
        super().__init__(
            CONFIG.sms_tr, siamese,
            early_stopping=CONFIG.sms_clbs['early_stopping'],
            model_checkpoint='siamese_checkpoint_fold{}.hdf5'.format(fold)
        )

    def __call__(self):
        return self.build_model()

    def build_model(self):
        # Single Branch Input
        branch_input = Input(shape=(CONFIG.enc_arc[-1]['units'] * 2,))

        # Single Branch Model
        branch_out = None
        for layer in CONFIG.sms_brn_arc:
            dropout = layer.pop('dropout')
            branch_out = Dense(**layer)(Dropout(dropout)(branch_input if branch_out is None else branch_out))
            layer['dropout'] = dropout
        branch_out = Dropout(CONFIG.sms_drp)(branch_input if branch_out is None else branch_out)

        model = Model(branch_input, branch_out)

        # Siamese Input
        input_a = Input(shape=(CONFIG.enc_arc[-1]['units'] * 2,))
        input_b = Input(shape=(CONFIG.enc_arc[-1]['units'] * 2,))

        # Siamese Branches
        branch_a = model(input_a)
        branch_b = model(input_b)

        # Merged Branches
        merged = getattr(layers, CONFIG.sms_mrg_md)([branch_a, branch_b])

        # Classifier
        output = None
        for layer in CONFIG.sms_clf_arc:
            dropout = layer.pop('dropout')
            output = Dense(**layer)(Dropout(dropout)(merged if output is None else output))
            layer['dropout'] = dropout
        output = Dense(1, activation='sigmoid')(Dropout(CONFIG.sms_drp)(output))

        # Classifier
        siamese = Model([input_a, input_b], output)
        siamese.summary()
        siamese.compile(**CONFIG.sms_ccfg)

        return siamese
