from keras.callbacks import TensorBoard

from utils.config import CONFIG


def rnn_tblogger(usr_num):
    return TensorBoard(
        log_dir=CONFIG.clbs['tensorboard']['log_dir_template'].format(
            usr_num=usr_num,
            bd='b' if CONFIG.bd_cell_type else '',
            ct=CONFIG.ct,
            earc='x'.join(map(str, CONFIG.enc_arc)),
            darc='x'.join(map(str, CONFIG.dec_arc)),
            epc=CONFIG.ae_tr_epochs
        ),
        histogram_freq=CONFIG.clbs['tensorboard']['histogram_freq'],
        write_images=CONFIG.clbs['tensorboard']['write_images']
    )
