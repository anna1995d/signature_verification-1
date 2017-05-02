import json
import logging
import os

from keras.callbacks import LambdaCallback, TensorBoard

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONIFG_PATH = os.path.join(PATH, 'configuration.json')
with open(CONIFG_PATH, 'r') as cf:
    CONFIG = json.load(cf)

logger = logging.getLogger(__name__)

elogger = LambdaCallback(
    on_epoch_end=lambda epoch, logs: logger.info('EPOCH #{epoch} end: loss: {loss}, accuracy: {accuracy}'.format(
        epoch=epoch, loss=logs['loss'], accuracy=logs['acc']
    ))
)

blogger = LambdaCallback(
    on_batch_end=lambda batch, logs: logger.info('BATCH #{batch} end: loss: {loss}, accuracy: {accuracy}'.format(
        batch=batch, loss=logs['loss'], accuracy=logs['acc']
    ))
)


def tblogger(usr_num):
    return TensorBoard(
        log_dir=CONFIG['tensorboard']['log_dir'].format(
            usr_num=usr_num,
            ct=CONFIG['autoencoder']['cell_type'],
            earc='x'.join(map(str, CONFIG['autoencoder']['encoder_architecture'])),
            darc='x'.join(map(str, CONFIG['autoencoder']['decoder_architecture'])),
            epc=CONFIG['autoencoder']['train_epochs']
        ),
        histogram_freq=CONFIG['tensorboard']['histogram_freq'],
        write_images=CONFIG['tensorboard']['write_images']
    )
