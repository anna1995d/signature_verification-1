import itertools

import numpy as np


class Data(object):
    @staticmethod
    def calculate_derivatives(d, smpl):
        if smpl:
            return np.concatenate(((d[1] - d[0]).reshape((1, -1)), d[1:] - d[:-1]))

        return np.concatenate((
            (d[1] - d[0]).reshape((1, -1)) + 2 * (d[2] - d[0]).reshape((1, -1)),
            (d[2] - d[0]).reshape((1, -1)) + 2 * (d[3] - d[0]).reshape((1, -1)),
            (d[3:-1] - d[1:-3]) + 2 * (d[3:-1] - d[1:-3]),
            (d[-1] - d[-3]).reshape((1, -1)) + 2 * (d[-1] - d[-4]).reshape((1, -1)),
            (d[-1] - d[-2]).reshape((1, -1)) + 2 * (d[-1] - d[-3]).reshape((1, -1)),
        )) / 10

    @staticmethod
    def extract(d):
        d = d - d[0]
        drv = Data.calculate_derivatives(d, smpl=True)
        t_n = np.arctan(drv[:, 1], drv[:, 0]).reshape((-1, 1))
        t_n[np.argwhere(np.isnan(t_n))] = 0
        v_n = np.sqrt(drv[:, 0] ** 2 + drv[:, 1] ** 2).reshape((-1, 1))
        dt_n = Data.calculate_derivatives(t_n, smpl=True)
        r_n = np.log(np.abs(v_n / (dt_n + np.finfo(np.float64).eps)) + np.finfo(np.float64).eps)
        dv_n = Data.calculate_derivatives(v_n, smpl=True)
        a_n = np.sqrt(dv_n ** 2 + (v_n * dt_n) ** 2)

        return np.concatenate((
            d,
            Data.calculate_derivatives(d, smpl=False),
            t_n,
            v_n,
            r_n,
            a_n,
            drv,
            dt_n,
            dv_n,
            Data.calculate_derivatives(np.concatenate((r_n, a_n), axis=1), smpl=True)
        ), axis=1)

    @staticmethod
    def normalize(d, nrm):
        if nrm == 'mvn' or nrm == 'mn':
            d = d - np.mean(d, axis=0)

        if nrm == 'mvn' or nrm == 'vn':
            d = d - np.std(d, axis=0, ddof=1)

        return d

    @staticmethod
    def extract_features(d, nrm):
        nd = Data.normalize(d, nrm)
        return Data.extract(nd)

    @staticmethod
    def extract_sample(smp_stp, nrm, path):
        with open(path, 'r') as f:
            d = np.reshape(f.read().split(), newshape=(-1, 3))[::smp_stp, :2].astype(np.float64)

        return Data.extract_features(d, nrm)

    @staticmethod
    def extract_genuine(smp_stp, nrm, usr, smp_cnt, path_temp):
        res = list()

        for i in range(smp_cnt):
            path = path_temp.format(user=usr, sample='{:02d}'.format(i + 1))
            res.append(Data.extract_sample(smp_stp=smp_stp, nrm=nrm, path=path))

        return res

    @staticmethod
    def extract_forged(smp_stp, nrm, usr, smp_cnt, frg_cnt, path_temp):
        res = list()

        for i in range(1, smp_cnt + 1):
            path = path_temp.format(
                user=usr,
                sample='{:02d}'.format((i + frg_cnt - 1) // frg_cnt),
                forger='{:03d}'.format(i % frg_cnt)
            )
            res.append(Data.extract_sample(smp_stp=smp_stp, nrm=nrm, path=path))

        return res

    def __init__(self, smp_stp, nrm, usr_cnt, gen_smp_cnt, frg_smp_cnt, frg_cnt, gen_path_temp, frg_path_temp):

        self.gen = [Data.extract_genuine(
            smp_stp=smp_stp,
            nrm=nrm,
            usr='{:03d}'.format(i),
            smp_cnt=gen_smp_cnt,
            path_temp=gen_path_temp
        ) for i in range(usr_cnt)]

        self.frg = [Data.extract_forged(
            smp_stp=smp_stp,
            nrm=nrm,
            usr='{:03d}'.format(i),
            smp_cnt=frg_smp_cnt,
            frg_cnt=frg_cnt,
            path_temp=frg_path_temp
        ) for i in range(usr_cnt)]

    def get_combinations(self, user, forged=False):
        d = self.frg[user] if forged else self.gen[user]
        return [_[0] for _ in itertools.product(d, d)], [_[1] for _ in itertools.product(d, d)]
