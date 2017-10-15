import logging

import numpy as np

from utils.config import CONFIG

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class Data(object):
    @staticmethod
    def calculate_derivatives(data, simple):
        if simple:
            return np.concatenate(((data[1] - data[0]).reshape((1, -1)), data[1:] - data[:-1]))
        else:
            return np.concatenate((
                (data[1] - data[0]).reshape((1, -1)) + 2 * (data[2] - data[0]).reshape((1, -1)),
                (data[2] - data[0]).reshape((1, -1)) + 2 * (data[3] - data[0]).reshape((1, -1)),
                (data[3:-1] - data[1:-3]) + 2 * (data[3:-1] - data[1:-3]),
                (data[-1] - data[-3]).reshape((1, -1)) + 2 * (data[-1] - data[-4]).reshape((1, -1)),
                (data[-1] - data[-2]).reshape((1, -1)) + 2 * (data[-1] - data[-3]).reshape((1, -1)),
            )) / 10

    @staticmethod
    def extract_features(data):
        drv_s = Data.calculate_derivatives(data, simple=True)
        drv = Data.calculate_derivatives(data, simple=False)
        t_n = np.arctan(drv_s[:, 1], drv_s[:, 0]).reshape((-1, 1))
        v_n = np.sqrt(drv_s[:, 0] ** 2 + drv_s[:, 1] ** 2).reshape((-1, 1))
        dt_n = Data.calculate_derivatives(t_n, simple=True)
        r_n = np.nan_to_num(np.log(np.abs(v_n / (dt_n + np.finfo(np.float32).eps)) + np.finfo(np.float32).eps))
        dv_n = Data.calculate_derivatives(v_n, simple=True)
        a_n = np.sqrt(dv_n ** 2 + (v_n * dt_n) ** 2)
        dar = Data.calculate_derivatives(np.concatenate((r_n, a_n), axis=1), simple=True)

        return np.concatenate((data, drv_s, drv, t_n, v_n, dt_n, r_n, dv_n, a_n, dar), axis=1)

    @staticmethod
    def normalize(data):
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0, ddof=1) + np.finfo(np.float32).eps)

    @staticmethod
    def extract_sample(dataset, writer, sample):
        data = dataset['U{wrt}S{smp}'.format(wrt=writer, smp=sample)].astype(np.float32)
        y = Data.normalize(Data.extract_features(data))
        iterator = range(CONFIG.win_rds, len(y) - CONFIG.win_rds, CONFIG.win_stp)
        x = np.concatenate([y[i - CONFIG.win_rds:i + CONFIG.win_rds + 1].reshape((1, -1)) for i in iterator], axis=0)
        return x, y[CONFIG.win_rds:len(y) - CONFIG.win_rds:CONFIG.win_stp]

    @staticmethod
    def extract_writer(dataset, writer, start=0, stop=CONFIG.gen_smp_cnt + CONFIG.frg_smp_cnt):
        xs, ys = list(), list()
        for sample in range(start, stop):
            x, y = Data.extract_sample(dataset, writer, sample)

            if len(y) > CONFIG.len_thr:
                logger.info('Ignore sequence: Writer #{wrt}, Length {len}'.format(wrt=writer, len=len(y)))
                continue

            xs.append(x[::CONFIG.smp_stp]), ys.append(y[::CONFIG.smp_stp])
        return xs, ys

    def __init__(self):
        dataset = np.load(CONFIG.dataset_path)

        self.gen_x, self.gen_y, self.frg_x, self.frg_y = list(), list(), list(), list()
        for writer in range(CONFIG.wrt_cnt):
            if writer % 10 == 0:
                logger.info('Loading data: Writer #{wrt}'.format(wrt=writer))

            writer += (CONFIG.gap - CONFIG.tr_wrt_cnt) if writer >= CONFIG.tr_wrt_cnt else 0

            x, y = Data.extract_writer(dataset, writer, stop=CONFIG.gen_smp_cnt)
            self.gen_x.append(x), self.gen_y.append(y)

            x, y = Data.extract_writer(dataset, writer, start=CONFIG.gen_smp_cnt)
            self.frg_x.append(x), self.frg_y.append(y)

        gen_max_len = max(map(len, np.concatenate(self.gen_y)))
        logger.info('Genuine max length: {gen_max_len}'.format(gen_max_len=gen_max_len))

        gen_min_len = min(map(len, np.concatenate(self.gen_y)))
        logger.info('Genuine min length: {gen_min_len}'.format(gen_min_len=gen_min_len))

        frg_max_len = max(map(len, np.concatenate(self.frg_y)))
        logger.info('Forged max length: {frg_max_len}'.format(frg_max_len=frg_max_len))

        frg_min_len = min(map(len, np.concatenate(self.frg_y)))
        logger.info('Forged min length: {frg_min_len}'.format(frg_min_len=frg_min_len))

        self.max_len = max(gen_max_len, frg_max_len)

    def get_train_data(self, writer):
        if len(self.gen_x[writer]) > 0 or len(self.frg_x[writer]) > 0:
            return np.array(list(zip(self.gen_x[writer], self.gen_y[writer])) +
                            list(zip(self.frg_x[writer], self.frg_y[writer]))).T
        else:
            logger.info('Writer with no sequence: writer #{wrt}'.format(wrt=writer))
            return None, None
