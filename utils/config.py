import errno
import logging
import os

import yaml
from keras import optimizers


class Configuration(object):
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
        self.ae_md = config['autoencoder']['mode']
        self.ae_tr = config['autoencoder']['train']

        self.ae_drp = config['autoencoder']['architecture']['global']['dropout']
        self.ae_mrg_md = config['autoencoder']['architecture']['global']['merge_mode']
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

        self.ae_clbs = config['autoencoder']['callbacks']

        # General Configuration
        self.spt_cnt = config['general']['split_count']
        self.tr_wrt_cnt = config['general']['train_writer_count']
        self.ref_smp_cnt = config['general']['reference_sample_count']
        self.clf_rpt_dgt = config['general']['classification_report_digits']
        self.dir_temp = config['general']['directory_template'].format(
            ct=self.ct,
            earc='x'.join(map(lambda x: str(x['units']), self.enc_arc)),
            darc='x'.join(map(lambda x: str(x['units']), self.dec_arc)),
            epc=self.ae_tr['epochs']
        )
        out_dir_temp = os.path.join(self.path, config['general']['output_directory_template'])

        # Data Configuration
        self.ftr = config['data']['reshaping']['features']
        self.smp_stp = config['data']['reshaping']['sampling_step']
        self.win_rds = config['data']['reshaping']['window_radius']
        self.win_stp = config['data']['reshaping']['window_step']
        self.len_thr = config['data']['reshaping']['length_threshold']

        self.wrt_cnt = config['data']['reading']['writer_count']
        self.gen_smp_cnt = config['data']['reading']['genuine_sample_count']
        self.frg_smp_cnt = config['data']['reading']['forged_sample_count']
        self.dataset_path = os.path.join(self.path, config['data']['reading']['dataset_path'])

        # Siamese Configuration
        self.sms_md = config['siamese']['mode']
        self.sms_tr = config['siamese']['train']
        self.sms_act = config['siamese']['activation']
        self.sms_ts_prb_thr = config['siamese']['test']['probability_threshold']
        self.sms_ts_acc_thr = config['siamese']['test']['accept_threshold']

        self.sms_drp = config['siamese']['architecture']['global']['dropout']
        self.sms_brn_arc = list(map(
            lambda x: x[1], sorted(list(config['siamese']['architecture']['before'].items()), key=lambda x: x[0])
        ))
        self.sms_clf_arc = list(map(
            lambda x: x[1], sorted(list(config['siamese']['architecture']['after'].items()), key=lambda x: x[0])
        ))
        self.sms_mrg_md = config['siamese']['merge_mode']

        self.sms_ccfg = config['siamese']['compile_config']
        self.sms_ccfg['optimizer'] = getattr(optimizers, self.sms_ccfg['optimizer']['name'])(
            **self.sms_ccfg['optimizer']['args']
        )

        self.sms_clbs = config['siamese']['callbacks']

        # Logger Configuration
        self.log_frm = config['logger']['log_format']
        self.log_fl = config['logger']['log_file'].format(dir=self.dir_temp)
        self.log_lvl = getattr(logging, config['logger']['log_level'].upper())
        self.configure_logger()

        # Export Configuration
        self.evaluation = config['export']['evaluation']
        self.out_dir = out_dir_temp.format(dir=self.dir_temp)


CONFIG = Configuration()
