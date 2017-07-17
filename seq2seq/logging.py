import logging

from keras.callbacks import LambdaCallback

logger = logging.getLogger(__name__)

elogger = LambdaCallback(
    on_epoch_end=lambda epoch, logs: logger.info(
        'EPOCH #{epoch}: loss: {loss}, acc: {acc}, val_loss: {val_loss}, val_acc: {val_acc}'.format(
            epoch=epoch, loss=logs['loss'], acc=logs['acc'], val_loss=logs['val_loss'], val_acc=logs['val_acc']
        )
    )
)

blogger = LambdaCallback(
    on_batch_end=lambda batch, logs: logger.info(
        'BATCH #{batch}: loss: {loss}, acc: {acc}'.format(batch=batch, loss=logs['loss'], acc=logs['acc'])
    )
)
