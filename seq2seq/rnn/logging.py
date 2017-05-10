import json
import os

from keras.callbacks import TensorBoard

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONIFG_PATH = os.path.join(PATH, 'configuration.json')
with open(CONIFG_PATH, 'r') as cf:
    CONFIG = json.load(cf)


def rnn_tblogger(usr_num):
    return TensorBoard(
        log_dir=CONFIG['tensorboard']['log_dir_template'].format(
            usr_num=usr_num,
            bd='b' if CONFIG['rnn']['autoencoder']['architecture']['bidirectional'] else '',
            ct=CONFIG['rnn']['autoencoder']['architecture']['cell_type'],
            earc='x'.join(map(str, CONFIG['rnn']['autoencoder']['architecture']['encoder'])),
            darc='x'.join(map(str, CONFIG['rnn']['autoencoder']['architecture']['decoder'])),
            epc=CONFIG['rnn']['autoencoder']['train']['epochs']
        ),
        histogram_freq=CONFIG['tensorboard']['histogram_freq'],
        write_images=CONFIG['tensorboard']['write_images']
    )
