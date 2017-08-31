import logging

from keras.callbacks import LambdaCallback

logger = logging.getLogger(__name__)


elogger = LambdaCallback(
    on_epoch_end=lambda batch, logs: logger.info(
        'EPOCH #{}: {}'.format(batch, ' '.join(['{}: {}'.format(k, v) for k, v in logs.items()]))
    )
)
