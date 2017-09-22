import errno
import logging
import os

import numpy as np
import yaml
from keras import optimizers


class Configuration(object):
    def configure_numpy(self):
        np.random.seed(self.rnd_sd)

    def configure_logger(self):
        if not os.path.exists(os.path.dirname(self.log_fl)):
            os.mkdir(os.path.dirname(self.log_fl))
        logging.basicConfig(filename=self.log_fl, level=self.log_lvl, format=self.log_frm)

    def __init__(self):
        self.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(self.path, 'configuration.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'configuration.yaml')
        with open(config_path, 'r') as cf:
            config = yaml.load(cf)

        # Autoencoder Configuration
        self.ae_tr = config['autoencoder']['train']

        self.enc_arc = list(map(
            lambda x: x[1], sorted(list(config['autoencoder']['architecture']['encoder'].items()), key=lambda x: x[0])
        ))
        self.dec_arc = list(map(
            lambda x: x[1], sorted(list(config['autoencoder']['architecture']['decoder'].items()), key=lambda x: x[0])
        ))
        self.ct = config['autoencoder']['architecture']['cell_type']

        self.ae_ccfg = config['autoencoder']['compile_config']
        self.ae_ccfg['optimizer'] = getattr(optimizers, self.ae_ccfg['optimizer']['name'])(
            **self.ae_ccfg['optimizer']['args']
        )

        self.clbs = config['autoencoder']['callbacks']

        # General Configuration
        self.rnd_sd = config['general']['random_seed']
        self.configure_numpy()
        self.dir_temp = config['general']['directory_template'].format(
            ct=self.ct,
            earc='x'.join(map(lambda x: str(x['units']), self.enc_arc)),
            darc='x'.join(map(lambda x: str(x['units']), self.dec_arc)),
            epc=self.ae_tr['epochs']
        )
        self.out_dir_temp = os.path.join(self.path, config['general']['output_directory_template'])

        # Data Configuration
        self.inp_dim = config['data']['reshaping']['input_dimension']
        self.smp_stp = config['data']['reshaping']['sampling_step']
        self.win_sze = config['data']['reshaping']['window_size']
        self.win_stp = config['data']['reshaping']['window_step']
        self.len_thr = config['data']['reshaping']['length_threshold']

        self.wrt_cnt = config['data']['reading']['writer_count']
        self.gen_smp_cnt = config['data']['reading']['genuine_sample_count']
        self.frg_smp_cnt = config['data']['reading']['forged_sample_count']
        self.dataset_path = os.path.join(self.path, config['data']['reading']['dataset_path'])

        # Classifiers Configuration
        self.ref_smp_cnt = config['classifiers']['reference_sample_count']
        self.clf_tr_wrt_cnt = config['classifiers']['train_writer_count']
        self.clf_ts_wrt_cnt = config['classifiers']['test_writer_count']
        self.knc = config['classifiers']['knc']
        self.svc = config['classifiers']['svc']

        # Logger Configuration
        self.log_frm = config['logger']['log_format']
        self.log_fl = config['logger']['log_file'].format(dir=self.dir_temp)
        self.log_lvl = getattr(logging, config['logger']['log_level'].upper())
        self.configure_logger()

        # Export Configuration
        self.csv = config['export']['csv']
        self.out_dir = self.out_dir_temp.format(dir=self.dir_temp)


CONFIG = Configuration()
