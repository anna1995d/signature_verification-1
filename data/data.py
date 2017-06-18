import itertools

import numpy as np

from utils.config import CONFIG


class Data(object):
    @staticmethod
    def calculate_derivatives(data, smpl):
        if smpl:
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
    def extract(data):
        drvs = Data.calculate_derivatives(data, smpl=True)
        drv = Data.calculate_derivatives(data, smpl=False)
        t_n = np.arctan(drvs[:, 1], drvs[:, 0]).reshape((-1, 1))
        v_n = np.sqrt(drvs[:, 0] ** 2 + drvs[:, 1] ** 2).reshape((-1, 1))
        dt_n = Data.calculate_derivatives(t_n, smpl=True)
        r_n = np.nan_to_num(np.log(np.abs(v_n / (dt_n + np.finfo(np.float64).eps)) + np.finfo(np.float64).eps))

        present = np.concatenate((data, drvs, drv, t_n, v_n, r_n), axis=1)
        past = np.concatenate((present[0, :].reshape((1, -1)), present[:-1, :]))
        future = np.concatenate((present[1:, :], present[-1, :].reshape((1, -1))))

        return np.concatenate((past, present, future), axis=1), present

    @staticmethod
    def normalize(data):
        data -= np.mean(data, axis=0) if 'm' in CONFIG.nrm else 0
        data -= np.std(data, axis=0, ddof=1) if 's' in CONFIG.nrm else 0
        data -= data[0]
        return data

    @staticmethod
    def extract_features(data):
        return Data.extract(Data.normalize(data))

    @staticmethod
    def extract_sample(path):
        with open(path, 'r') as f:
            raw_data = np.reshape(f.read().split(), newshape=(-1, CONFIG.ftr_cnt))[::CONFIG.smp_stp, :2].astype(
                np.float64
            )

        data = np.concatenate(
            [np.roll(raw_data, -ln, axis=0) for ln in range(CONFIG.rl_win_sz)], axis=1
        )[:(1 - CONFIG.rl_win_sz) or None:CONFIG.rl_win_stp]

        return Data.extract_features(data)

    @staticmethod
    def extract_genuine(usr_num):
        gen_x, gen_y = list(), list()

        for smp in range(CONFIG.gen_smp_cnt):
            x, y = Data.extract_sample(path=CONFIG.gen_path_temp.format(user=usr_num, sample=smp + 1))
            gen_x.append(x)
            gen_y.append(y)

        return gen_x, gen_y

    @staticmethod
    def extract_forged(usr_num):
        frg_x, frg_y = list(), list()

        for smp in range(CONFIG.gen_smp_cnt, CONFIG.gen_smp_cnt + CONFIG.frg_smp_cnt):
            x, y = Data.extract_sample(path=CONFIG.frg_path_temp.format(user=usr_num, sample=smp + 1))
            frg_x.append(x)
            frg_y.append(y)

        return frg_x, frg_y

    def __init__(self):
        self.gen_x, self.gen_y, self.frg_x, self.frg_y = list(), list(), list(), list()
        for usr_num in range(1, CONFIG.usr_cnt + 1):
            x, y = Data.extract_genuine(usr_num=usr_num)
            self.gen_x.append(x)
            self.gen_y.append(y)

            x, y = Data.extract_forged(usr_num=usr_num)
            self.frg_x.append(x)
            self.frg_y.append(y)

        self.gen_max_len = max(map(lambda x: max(map(lambda y: len(y), x)), self.gen_x))

    def get_genuine_combinations(self, usr_num):
        return np.array(
            list(itertools.product(self.gen_x[usr_num][:CONFIG.ae_smp_cnt], self.gen_y[usr_num][:CONFIG.ae_smp_cnt]))
        ).T
