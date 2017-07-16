import os
import errno
import json
import logging

import numpy as np


class Configuration(object):
    def configure_numpy(self):
        np.random.seed(self.rnd_sd)

    def configure_logger(self):
        if not os.path.exists(os.path.dirname(self.log_fl)):
            os.mkdir(os.path.dirname(self.log_fl))
        logging.basicConfig(filename=self.log_fl, level=self.log_lvl, format=self.log_frm)

    def __init__(self):
        self.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(self.path, 'configuration.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'configuration.json')
        with open(config_path, 'r') as cf:
            config = json.load(cf)

        # Autoencoder Configuration
        self.ae_btch_sz = config['rnn']['autoencoder']['train']['batch_size']
        self.ae_tr_epochs = config['rnn']['autoencoder']['train']['epochs']
        self.ae_smp_cnt = config['rnn']['autoencoder']['train']['sampling_count']

        self.enc_arc = config['rnn']['autoencoder']['architecture']['encoder']
        self.dec_arc = config['rnn']['autoencoder']['architecture']['decoder']
        self.ct = config['rnn']['autoencoder']['architecture']['cell_type']

        self.ae_ccfg = config['rnn']['autoencoder']['compile_config']
        self.ae_lcfg = config['rnn']['autoencoder']['layers_config']

        self.clbs = config['rnn']['autoencoder']['callbacks']

        # General Configuration
        self.rnd_sd = config['general']['random_seed']
        self.configure_numpy()
        self.verbose = config['general']['verbose']
        self.dir_temp = config['general']['directory_template'].format(
            ct=self.ct,
            earc='x'.join(map(str, self.enc_arc)),
            darc='x'.join(map(str, self.dec_arc)),
            epc=self.ae_tr_epochs
        )
        self.out_dir_temp = os.path.join(self.path, config['general']['output_directory_template'])

        # Data Configuration
        self.inp_dim = config['data']['reshaping']['input_dimension']
        self.smp_stp = config['data']['reshaping']['sampling_step']

        self.usr_cnt = config['data']['reading']['user_count']
        self.gen_smp_cnt = config['data']['reading']['genuine_sample_count']
        self.frg_smp_cnt = config['data']['reading']['forged_sample_count']
        self.sig_path_temp = os.path.join(self.path, config['data']['reading']['signature_path_template'])
        self.ftr_cnt = config['data']['reading']['feature_count']

        # SVC Configuration
        self.svc_smp_cnt = config['svc']['reference_sample_count']
        self.svc_tr_usr_cnt = config['svc']['train_user_count']
        self.svc_ts_usr_cnt = config['svc']['test_user_count']

        # Logger Configuration
        self.log_frm = config['logger']['log_format']
        self.log_fl = config['logger']['log_file'].format(dir=self.dir_temp)
        self.log_lvl = getattr(logging, config['logger']['log_level'].upper())
        self.configure_logger()

        # Export Configuration
        self.mdl_save_temp = config['export']['model_save_template']
        self.svc_csv_fns = config['export']['svc_csv_fieldnames']
        self.out_dir = self.out_dir_temp.format(dir=self.dir_temp)


CONFIG = Configuration()
