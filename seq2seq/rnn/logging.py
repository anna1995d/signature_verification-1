from keras.callbacks import TensorBoard

from utils.config import CONFIG


def rnn_tblogger():
    return TensorBoard(log_dir=CONFIG.clbs['tensorboard']['log_dir_template'].format(dir=CONFIG.dir_temp))
