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

tblogger = TensorBoard(log_dir=CONFIG['logger']['log_dir'], histogram_freq=1, write_images=True)
