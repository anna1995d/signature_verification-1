import logging

from keras.callbacks import LambdaCallback

logger = logging.getLogger(__name__)

klogger = LambdaCallback(
    on_epoch_end=lambda epoch, logs: logger.info('EPOCH #{epoch} end: loss: {loss}, accuracy: {accuracy}'.format(
        epoch=epoch, loss=logs['loss'], accuracy=logs['acc']
    )),
    on_batch_end=lambda batch, logs: logger.info('BATCH #{batch} end: loss: {loss}, accuracy: {accuracy}'.format(
        batch=batch, loss=logs['loss'], accuracy=logs['acc']
    ))
)
