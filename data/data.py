import numpy as np

from utils.config import CONFIG


class Data(object):
    @staticmethod
    def calculate_derivatives(data, smp):
        if smp:
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
        drv_s = Data.calculate_derivatives(data, smp=True)
        drv = Data.calculate_derivatives(data, smp=False)
        t_n = np.arctan(drv_s[:, 1], drv_s[:, 0]).reshape((-1, 1))
        v_n = np.sqrt(drv_s[:, 0] ** 2 + drv_s[:, 1] ** 2).reshape((-1, 1))
        dt_n = Data.calculate_derivatives(t_n, smp=True)
        r_n = np.nan_to_num(np.log(np.abs(v_n / (dt_n + np.finfo(np.float64).eps)) + np.finfo(np.float64).eps))
        dv_n = Data.calculate_derivatives(v_n, smp=True)
        a_n = np.sqrt(dv_n ** 2 + (v_n * dt_n) ** 2)
        dar = Data.calculate_derivatives(np.concatenate((r_n, a_n), axis=1), smp=True)

        return np.concatenate((data, drv_s, drv, t_n, v_n, dt_n, r_n, dv_n, a_n, dar), axis=1)

    @staticmethod
    def normalize(data):
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0, ddof=1) + np.finfo(np.float64).eps)

    @staticmethod
    def extract_sample(path):
        with open(path, 'r') as f:
            data = np.reshape(f.read().split(), newshape=(-1, CONFIG.ftr_cnt))[::CONFIG.smp_stp, :2].astype(np.float64)
        features = Data.normalize(Data.extract_features(data))
        flatten_features = features.flatten()
        step = CONFIG.win_stp * CONFIG.inp_dim
        window = CONFIG.win_sze * CONFIG.inp_dim
        iterator = range(0, flatten_features.shape[0] - window + 1, step)
        return np.concatenate([flatten_features[i:i + window].reshape((1, -1)) for i in iterator], axis=0), features

    @staticmethod
    def extract_user(usr_num, start=0, stop=CONFIG.gen_smp_cnt + CONFIG.frg_smp_cnt):
        xs, ys = list(), list()
        for smp in range(start, stop):
            x, y = Data.extract_sample(path=CONFIG.sig_path_temp.format(user=usr_num, sample=smp + 1))
            xs.append(x), ys.append(y)
        return xs, ys

    def __init__(self):
        self.gen_x, self.gen_y, self.frg_x, self.frg_y = list(), list(), list(), list()
        for usr_num in range(1, CONFIG.usr_cnt + 1):
            x, y = Data.extract_user(usr_num=usr_num, stop=CONFIG.gen_smp_cnt)
            self.gen_x.append(x), self.gen_y.append(y)

            x, y = Data.extract_user(usr_num=usr_num, start=CONFIG.gen_smp_cnt)
            self.frg_x.append(x), self.frg_y.append(y)

        self.gen_max_len = max(map(lambda tmp_x: max(map(lambda tmp_y: len(tmp_y), tmp_x)), self.gen_x))

    def get_genuine_combinations(self, usr_num):
        return np.array(list(zip(self.gen_x[usr_num], self.gen_y[usr_num]))).T
