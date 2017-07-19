import logging

from keras.callbacks import LambdaCallback, TensorBoard

from utils.config import CONFIG

logger = logging.getLogger(__name__)


def rnn_tblogger():
    return TensorBoard(log_dir=CONFIG.clbs['tensorboard']['log_dir_template'].format(dir=CONFIG.dir_temp))

elogger = LambdaCallback(
    on_epoch_end=lambda epoch, logs: logger.info(
        'EPOCH #{epoch}: loss: {loss}, acc: {acc}'.format(epoch=epoch, loss=logs['loss'], acc=logs['acc'])
    )
)

blogger = LambdaCallback(
    on_batch_end=lambda batch, logs: logger.info(
        'BATCH #{batch}: loss: {loss}, acc: {acc}'.format(batch=batch, loss=logs['loss'], acc=logs['acc'])
    )
)
